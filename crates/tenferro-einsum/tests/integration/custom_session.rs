//! Customization through the public session and provider surfaces (#1938 D7/D8).
//!
//! A custom GEMM provider installed on a `CpuBackend` is reached by standard
//! tensordot and einsum, and a user session wrapping a standard CPU session can
//! override one operation while delegating the rest, with its native-service
//! exposure under its own control.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tenferro_cpu::provider::{
    CpuExecutionContext, CpuGemmProvider, CpuGemmRequest, CpuGroupedGemmRequest,
    CpuProviderOutcome, FaerGemmProvider, StridedLayoutTransformProvider,
};
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuProviderBundle};
use tenferro_einsum::{TensorDotAxes, TensorEinsumExt, TensorTensordotExt};
use tenferro_tensor::{
    BackendCachedDot, BackendSession, BackendSessionHost, CompareDir, DType, DotGeneralConfig,
    ElementwiseReadOp, GatherConfig, NativeSessionRef, PadConfig, ScatterConfig, SliceConfig,
    Tensor, TensorAnalytic, TensorBuffer, TensorDeviceTransfer, TensorDot, TensorElementwise,
    TensorFusion, TensorIndexing, TensorRead, TensorReduction, TensorStructural, TensorWrite,
};

/// A user GEMM that delegates the arithmetic to faer and counts its calls.
#[derive(Debug, Default)]
struct CountingGemm {
    calls: Arc<AtomicUsize>,
}

impl CpuGemmProvider for CountingGemm {
    fn execution_capabilities(&self) -> tenferro_cpu::CpuProviderExecutionCapabilities {
        FaerGemmProvider.execution_capabilities()
    }

    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        FaerGemmProvider.gemm(context, request)
    }

    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        FaerGemmProvider.strided_batched_gemm(context, request)
    }

    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> tenferro_tensor::Result<CpuProviderOutcome> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        FaerGemmProvider.grouped_gemm(context, request)
    }
}

fn matrices() -> (Tensor, Tensor) {
    (
        Tensor::from_vec_col_major(vec![2, 3], vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap(),
        Tensor::from_vec_col_major(vec![3, 2], vec![1.0_f64, 0.0, 2.0, 0.0, 1.0, 1.0]).unwrap(),
    )
}

// Column-major [[1, 3, 5], [2, 4, 6]] times [[1, 0], [0, 1], [2, 1]].
const PRODUCT: [f64; 4] = [11.0, 14.0, 8.0, 10.0];

#[test]
fn custom_gemm_provider_is_reached_by_standard_tensordot_and_einsum() {
    let calls = Arc::new(AtomicUsize::new(0));
    let bundle = CpuProviderBundle::custom_builder()
        .gemm_provider(Arc::new(CountingGemm {
            calls: Arc::clone(&calls),
        }))
        .layout_transform_provider(Arc::new(StridedLayoutTransformProvider))
        .build()
        .unwrap();
    let mut backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_provider_bundle(bundle)
        .unwrap();
    let (lhs, rhs) = matrices();

    let dotted = backend
        .with_backend_session(|session| lhs.tensordot(&rhs, TensorDotAxes::Count(1), session))
        .unwrap()
        .unwrap();
    assert_eq!(dotted.as_slice::<f64>().unwrap(), &PRODUCT);
    let after_tensordot = calls.load(Ordering::Relaxed);
    assert!(after_tensordot >= 1);

    let summed = backend
        .with_backend_session(|session| [&lhs, &rhs].einsum("ij,jk->ik", session))
        .unwrap()
        .unwrap();
    assert_eq!(summed.as_slice::<f64>().unwrap(), &PRODUCT);
    assert!(calls.load(Ordering::Relaxed) > after_tensordot);
}

/// A user session that overrides `dot_general_read` and delegates everything
/// else to the standard session it wraps.
struct CountingDotSession<'s> {
    inner: &'s mut dyn BackendSession,
    dot_calls: usize,
    forward_native: bool,
}

macro_rules! forward {
    ($($name:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        $(fn $name(&mut self, $($arg: $ty),*) -> $ret {
            self.inner.$name($($arg),*)
        })*
    };
}

impl TensorElementwise for CountingDotSession<'_> {
    forward! {
        elementwise_read_into(op: ElementwiseReadOp, inputs: &[TensorRead<'_>], out: TensorWrite<'_>) -> tenferro_tensor::Result<()>;
        add_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        sub_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        mul_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        neg_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        conj_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        div_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        abs_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        sign_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        maximum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        minimum_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        compare_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>, dir: &CompareDir) -> tenferro_tensor::Result<Tensor>;
        select_read(pred: TensorRead<'_>, on_true: TensorRead<'_>, on_false: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        clamp_read(input: TensorRead<'_>, lower: TensorRead<'_>, upper: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorAnalytic for CountingDotSession<'_> {
    forward! {
        exp_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        log_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        sin_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        cos_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        tanh_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        sqrt_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        rsqrt_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        pow_read(lhs: TensorRead<'_>, rhs: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        expm1_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        log1p_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorStructural for CountingDotSession<'_> {
    forward! {
        to_contiguous_read(input: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        copy_read_into(src: TensorRead<'_>, dst: TensorWrite<'_>) -> tenferro_tensor::Result<()>;
        transpose_read(input: TensorRead<'_>, perm: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reshape_read(input: TensorRead<'_>, shape: &[usize]) -> tenferro_tensor::Result<Tensor>;
        broadcast_in_dim_read(input: TensorRead<'_>, shape: &[usize], dims: &[usize]) -> tenferro_tensor::Result<Tensor>;
        cast(input: &Tensor, to: DType) -> tenferro_tensor::Result<Tensor>;
        extract_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        embed_diagonal(input: &Tensor, axis_a: usize, axis_b: usize) -> tenferro_tensor::Result<Tensor>;
        tril(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
        triu(input: &Tensor, k: i64) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorReduction for CountingDotSession<'_> {
    forward! {
        reduce_sum_read(input: TensorRead<'_>, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_prod_read(input: TensorRead<'_>, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_max_read(input: TensorRead<'_>, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        reduce_min_read(input: TensorRead<'_>, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorIndexing for CountingDotSession<'_> {
    forward! {
        gather(operand: &Tensor, start_indices: &Tensor, config: &GatherConfig) -> tenferro_tensor::Result<Tensor>;
        scatter(operand: &Tensor, scatter_indices: &Tensor, updates: &Tensor, config: &ScatterConfig) -> tenferro_tensor::Result<Tensor>;
        slice(input: &Tensor, config: &SliceConfig) -> tenferro_tensor::Result<Tensor>;
        dynamic_slice(input: &Tensor, starts: &Tensor, slice_sizes: &[usize]) -> tenferro_tensor::Result<Tensor>;
        dynamic_update_slice(operand: &Tensor, update: &Tensor, starts: &Tensor) -> tenferro_tensor::Result<Tensor>;
        pad(input: &Tensor, config: &PadConfig) -> tenferro_tensor::Result<Tensor>;
        concatenate(inputs: &[&Tensor], axis: usize) -> tenferro_tensor::Result<Tensor>;
        reverse(input: &Tensor, axes: &[usize]) -> tenferro_tensor::Result<Tensor>;
    }
}

impl TensorDot for CountingDotSession<'_> {
    fn dot_general_read(
        &mut self,
        lhs: TensorRead<'_>,
        rhs: TensorRead<'_>,
        config: &DotGeneralConfig,
    ) -> tenferro_tensor::Result<Tensor> {
        // The override: count, then use the standard implementation.
        self.dot_calls += 1;
        self.inner.dot_general_read(lhs, rhs, config)
    }
}

impl TensorFusion for CountingDotSession<'_> {}
impl TensorBuffer for CountingDotSession<'_> {}

impl TensorDeviceTransfer for CountingDotSession<'_> {
    forward! {
        download_to_host(tensor: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
        upload_host_tensor(tensor: TensorRead<'_>) -> tenferro_tensor::Result<Tensor>;
    }
}

impl tenferro_tensor::BackendRuntimeCache for CountingDotSession<'_> {
    type RuntimeCache = ();
}
impl BackendCachedDot for CountingDotSession<'_> {}
impl tenferro_tensor::SessionCachedDot for CountingDotSession<'_> {}

impl BackendSession for CountingDotSession<'_> {
    fn native_session(&mut self) -> Option<NativeSessionRef<'_>> {
        // Forwarding hands operation families the wrapped CPU session's native
        // services, which bypass the override; keep the default unless that is
        // intended.
        if self.forward_native {
            self.inner.native_session()
        } else {
            None
        }
    }
}

#[test]
fn a_wrapping_session_overrides_one_operation_and_controls_native_exposure() {
    let mut backend = CpuBackend::with_threads(1).unwrap();
    let (lhs, rhs) = matrices();

    for forward_native in [false, true] {
        let (product, dot_calls, native_visible) = backend
            .with_backend_session(|session| {
                let mut wrapper = CountingDotSession {
                    inner: session,
                    dot_calls: 0,
                    forward_native,
                };
                let product = [&lhs, &rhs].einsum("ij,jk->ik", &mut wrapper).unwrap();
                let native_visible = with_cpu_exec_session(&mut wrapper, |_| ()).is_some();
                (product, wrapper.dot_calls, native_visible)
            })
            .unwrap();

        assert_eq!(product.as_slice::<f64>().unwrap(), &PRODUCT);
        assert!(dot_calls >= 1, "standard einsum must reach the override");
        assert_eq!(native_visible, forward_native);
    }
}
