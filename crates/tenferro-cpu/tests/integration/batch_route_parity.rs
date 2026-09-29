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

fn spy_backend(
    threads: usize,
    policy: Option<tenferro_cpu::CpuBatchPolicy>,
) -> (Arc<Spy>, CpuBackend) {
    let spy = Arc::new(Spy::default());
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(spy.clone())
        .layout_transform_provider(Arc::new(StridedLayoutTransformProvider))
        .build()
        .unwrap();
    let mut cpu = CpuBackend::with_threads(threads)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    if let Some(policy) = policy {
        cpu = cpu.with_batch_policy(policy);
    }
    (spy, cpu)
}

fn batched_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    }
}

/// A 4x4x4 x 64 batch is below the default 8 us per-lane cost cutoff.
fn short_batch() -> Tensor {
    Tensor::from_vec_col_major([4, 4, 64], vec![1.0_f64; 16 * 64]).unwrap()
}

fn lane_calls(calls: &[(&'static str, bool, usize)]) -> usize {
    calls.iter().filter(|call| call.1).count()
}

/// #1946 F3: the lane cost model is part of the effective batch policy, with
/// backend default < scoped override precedence.
#[test]
fn lane_cost_model_follows_the_effective_policy() {
    use tenferro_cpu::{with_batch_policy, CpuBatchPolicy, CpuBatchStrategy, CpuBatchThresholds};

    let eager_split = CpuBatchPolicy::new(CpuBatchStrategy::Auto)
        .with_thresholds(CpuBatchThresholds::default().with_lane_min_work_ns(0));
    let a = short_batch();
    let config = batched_config();

    // Default policy: the short batch stays one provider call.
    let (spy, mut cpu) = spy_backend(4, None);
    cpu.with_backend_session(|session| {
        session
            .dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                &config,
            )
            .unwrap();
    })
    .unwrap();
    assert_eq!(
        lane_calls(&spy.take()),
        0,
        "default: no split below the cost cutoff"
    );

    // Backend default override: the same batch splits.
    let (spy, mut cpu) = spy_backend(4, Some(eager_split));
    cpu.with_backend_session(|session| {
        session
            .dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                &config,
            )
            .unwrap();
    })
    .unwrap();
    assert_eq!(
        lane_calls(&spy.take()),
        4,
        "backend default override splits"
    );

    // Scoped override on a default backend applies inside the scope only.
    let (spy, mut cpu) = spy_backend(4, None);
    cpu.with_backend_session(|session| {
        with_batch_policy(session, eager_split, |session| {
            session
                .dot_general_read(
                    TensorRead::from_tensor(&a),
                    TensorRead::from_tensor(&a),
                    &config,
                )
                .unwrap();
        })
        .unwrap();
        assert_eq!(lane_calls(&spy.take()), 4, "scoped override splits");
        session
            .dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                &config,
            )
            .unwrap();
        assert_eq!(
            lane_calls(&spy.take()),
            0,
            "the override ends with its scope"
        );
    })
    .unwrap();
}

/// #1946 F3: a forced OuterParallel splits a strided batch on both routes,
/// regardless of the cost model, and is a typed error where no lanes exist.
#[test]
fn forced_outer_parallel_splits_strided_batches_or_fails_typed() {
    use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

    let forced = CpuBatchPolicy::new(CpuBatchStrategy::OuterParallel);
    let a = short_batch();
    let config = batched_config();
    let (spy, mut cpu) = spy_backend(4, Some(forced));
    let mut out = Tensor::from_vec_col_major([4, 4, 64], vec![0.0_f64; 16 * 64]).unwrap();
    cpu.with_backend_session(|session| {
        let fresh = session
            .dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                &config,
            )
            .unwrap();
        assert_eq!(fresh.as_slice::<f64>().unwrap(), vec![4.0; 16 * 64]);
        let allocated = spy.take();
        session
            .dot_general_read_into(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                &config,
                TensorWrite::from_tensor(&mut out),
            )
            .unwrap();
        assert_eq!(out.as_slice::<f64>().unwrap(), vec![4.0; 16 * 64]);
        for (route, calls) in [("allocating", allocated), ("into", spy.take())] {
            assert_eq!(lane_calls(&calls), 4, "{route}: {calls:?}");
            assert_eq!(
                calls.iter().map(|call| call.2).sum::<usize>(),
                64,
                "{route}"
            );
        }
    })
    .unwrap();

    let (spy, mut cpu) = spy_backend(1, Some(forced));
    let error = cpu
        .with_backend_session(|session| {
            session.dot_general_read(
                TensorRead::from_tensor(&a),
                TensorRead::from_tensor(&a),
                &config,
            )
        })
        .unwrap()
        .unwrap_err();
    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Unsupported,
        "{error}"
    );
    assert!(error.to_string().contains("OuterParallel"), "{error}");
    assert!(
        spy.take().is_empty(),
        "no provider call before the typed error"
    );
}
