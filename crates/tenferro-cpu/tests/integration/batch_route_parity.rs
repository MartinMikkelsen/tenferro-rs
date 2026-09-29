//! #1946 F2: an allocating batched dot and its `_into` twin take the same Auto
//! lane decision. The spy records every provider call and whether it ran on an
//! outer fan-out lane, and forwards unchanged to the standard faer provider.

use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex};

use tenferro_cpu::provider::{
    CpuGemmProvider, CpuGemmRequest, CpuGemmUninitRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, CpuUninitGemmProvider, FaerGemmProvider, StridedLayoutTransformProvider,
};
use tenferro_cpu::{
    CpuBackend, CpuExecutionContext, CpuProviderBundle, CpuProviderExecutionCapabilities,
};
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead, TensorWrite};

#[derive(Debug, Default)]
struct Spy(Mutex<Vec<(&'static str, bool, usize)>>);

impl Spy {
    fn take(&self) -> Vec<(&'static str, bool, usize)> {
        std::mem::take(&mut *self.0.lock().unwrap())
    }
}

impl CpuGemmProvider for Spy {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        FaerGemmProvider.execution_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.0.lock().unwrap().push((
            "initialized",
            context.is_outer_fan_out_lane(),
            request.batch_count(),
        ));
        FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        FaerGemmProvider.grouped_gemm(context, request)
    }

    fn uninit_provider(&self) -> Option<&dyn CpuUninitGemmProvider> {
        Some(self)
    }
}

// SAFETY: forwards the full-overwrite contract unchanged to the standard
// provider, which implements it.
unsafe impl CpuUninitGemmProvider for Spy {
    unsafe fn gemm_into_uninit(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmUninitRequest<'_, '_>,
        output: &mut [MaybeUninit<u8>],
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.0.lock().unwrap().push((
            "uninitialized",
            context.is_outer_fan_out_lane(),
            request.batch_count(),
        ));
        // SAFETY: the caller's contract is forwarded unchanged.
        unsafe { FaerGemmProvider.gemm_into_uninit(context, request, output) }
    }
}

#[test]
fn allocating_and_into_batched_dots_take_the_same_lane_decision() {
    let spy = Arc::new(Spy::default());
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(spy.clone())
        .layout_transform_provider(Arc::new(StridedLayoutTransformProvider))
        .build()
        .unwrap();
    let mut cpu = CpuBackend::with_threads(4)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    let a = Tensor::from_vec_col_major([4, 4, 1024], vec![1.0_f64; 16 * 1024]).unwrap();
    let mut out = Tensor::from_vec_col_major([4, 4, 1024], vec![0.0_f64; 16 * 1024]).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    };
    let (allocated, into) = cpu
        .with_backend_session(|session| {
            let fresh = session
                .dot_general_read(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&a),
                    &config,
                )
                .unwrap();
            assert_eq!(fresh.as_slice::<f64>().unwrap(), vec![4.0; 16 * 1024]);
            let allocated = spy.take();
            session
                .dot_general_read_into(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&a),
                    &config,
                    TensorWrite::from_tensor(&mut out),
                )
                .unwrap();
            assert_eq!(out.as_slice::<f64>().unwrap(), vec![4.0; 16 * 1024]);
            (allocated, spy.take())
        })
        .unwrap();
    for (route, calls) in [("allocating", &allocated), ("into", &into)] {
        assert!(
            calls.len() > 1 && calls.iter().all(|call| call.1),
            "{route}: Auto must split the batch over outer lanes, saw {calls:?}"
        );
        assert_eq!(
            calls.iter().map(|call| call.2).sum::<usize>(),
            1024,
            "{route}: the lane chunks must cover the batch exactly once"
        );
    }
    assert!(allocated.iter().all(|call| call.0 == "uninitialized"));
    assert!(into.iter().all(|call| call.0 == "initialized"));
}
