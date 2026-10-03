#![deny(missing_docs)]

//! Optional tprims-backed GEMM and `dot_general` providers for `tenferro-cpu`.
//!
//! [tprims](https://github.com/tensor4all/tprims-rs) takes an explicit
//! execution context that borrows a Rayon pool. [`TprimsProvider`] builds one
//! from the provider's [`CpuExecutionContext`]: the pool of the inner parallel
//! region ([`CpuExecutionContext::rayon_pool`]) with the context's thread
//! budget, or serial execution otherwise. Everything it does not handle is
//! reported as unsupported before any output is written, so the selected
//! `tenferro-cpu` backend runs it. That includes all linear algebra: tprims no
//! longer provides Cholesky, QR, SVD, eigh, solve or `trsm`, so
//! `tenferro-linalg`'s built-in kernels run them.
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//! use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuProviderBundle};
//! use tenferro_cpu_tprims::TprimsProvider;
//!
//! let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
//!     .gemm_provider(Arc::new(TprimsProvider::new()))
//!     .prefer_general_contraction_provider(Arc::new(TprimsProvider::new()));
//! let backend = CpuBackend::new().with_provider_bundle(builder.build()?)?;
//! # let _ = backend;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use num_complex::{Complex32, Complex64};
use std::collections::HashMap;

use strided_view::{StridedView, StridedViewMut};
use tenferro_cpu::provider::{
    CpuBatchedMatrixLayout, CpuDotGeneralRequest, CpuExecutionContext, CpuGemmProvider,
    CpuGemmRequest, CpuGeneralContractionProvider, CpuGroupedGemmRequest, CpuOperand,
    CpuProviderOutcome, CpuProviderUnsupported, CpuVendorBatch,
};
use tenferro_cpu::{CpuPlacementControl, CpuProviderExecutionCapabilities, CpuThreadCountControl};
use tenferro_tensor::{
    col_major_strides, ContractionScalar, DType, Error, Result, TensorRead, TensorScalar,
    TensorView, TensorViewMut, TensorWrite, TypedTensorView, TypedTensorViewMut,
};
use tprims_contract::api::{
    AccumulationSource, DotGeneral, LayoutSpec, Op, OperandSpec, Problem, Scalar,
};
use tprims_contract::{contract_batched, BatchItem, Plan, PlanConfig};
use tprims_exec::{Exec, Pool};

/// tprims implementation of `tenferro-cpu`'s GEMM and general-contraction
/// provider slots.
///
/// # Examples
///
/// ```
/// use tenferro_cpu::provider::CpuGemmProvider;
/// use tenferro_cpu_tprims::TprimsProvider;
/// let provider: &dyn CpuGemmProvider = &TprimsProvider::new();
/// let _ = provider.execution_capabilities();
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct TprimsProvider;

impl TprimsProvider {
    /// Create the provider.
    ///
    /// # Examples
    ///
    /// ```
    /// let _ = tenferro_cpu_tprims::TprimsProvider::new();
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

/// Parallel work runs on the engine's workers, within the per-call budget.
fn capabilities() -> CpuProviderExecutionCapabilities {
    CpuProviderExecutionCapabilities {
        thread_count: CpuThreadCountControl::PerCallUpperBound,
        placement: CpuPlacementControl::EngineWorkers,
        worker_local_sequential: true,
        accepts_sequential: true,
        accepts_outer: true,
        accepts_inner: true,
    }
}

/// Run `f` with the tprims execution context for `ctx`: the inner region's
/// pool bounded by the thread budget, or serial.
fn with_exec<R>(ctx: &CpuExecutionContext<'_>, f: impl FnOnce(&Exec<'_>) -> R) -> R {
    match ctx.rayon_pool() {
        Some(pool) => {
            let pool = Pool::borrow(pool);
            let exec = Exec::rayon(&pool)
                .with_budget(ctx.thread_budget().get())
                .unwrap_or(Exec::serial());
            f(&exec)
        }
        None => f(&Exec::serial()),
    }
}

/// The four element types tprims supports, with their tenferro views.
trait Elem: TensorScalar + Scalar {
    fn scalar(s: ContractionScalar) -> Option<Self>;
    fn view<'b, 'a>(v: &'b TensorView<'a>) -> Option<&'b TypedTensorView<'a, Self>>;
    fn view_mut<'b, 'a>(
        v: &'b mut TensorViewMut<'a>,
    ) -> Option<&'b mut TypedTensorViewMut<'a, Self>>;
}

macro_rules! elem {
    ($t:ty, $v:ident) => {
        impl Elem for $t {
            fn scalar(s: ContractionScalar) -> Option<Self> {
                match s {
                    ContractionScalar::$v(x) => Some(x),
                    _ => None,
                }
            }
            fn view<'b, 'a>(v: &'b TensorView<'a>) -> Option<&'b TypedTensorView<'a, Self>> {
                match v {
                    TensorView::$v(x) => Some(x),
                    _ => None,
                }
            }
            fn view_mut<'b, 'a>(
                v: &'b mut TensorViewMut<'a>,
            ) -> Option<&'b mut TypedTensorViewMut<'a, Self>> {
                match v {
                    TensorViewMut::$v(x) => Some(x),
                    _ => None,
                }
            }
        }
    };
}
elem!(f32, F32);
elem!(f64, F64);
elem!(Complex32, C32);
elem!(Complex64, C64);

/// A read operand: its whole backing storage and the layout of the tensor in
/// it (element strides and offset).
struct In<'a, T> {
    data: &'a [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

fn read<'a, T: Elem>(r: &'a TensorRead<'_>) -> Result<Option<In<'a, T>>> {
    Ok(match r {
        TensorRead::Tensor(t) => match t.as_typed::<T>() {
            Some(t) => Some(In {
                data: t.host_data()?,
                strides: col_major_strides(t.shape())?,
                shape: t.shape().to_vec(),
                offset: 0,
            }),
            None => None,
        },
        TensorRead::View(v) => match T::view(v) {
            Some(v) => Some(In {
                data: v.host_storage()?,
                shape: v.shape().to_vec(),
                strides: v.strides().to_vec(),
                offset: v.offset(),
            }),
            None => None,
        },
    })
}

/// The written operand, as [`In`].
struct Out<'a, T> {
    data: &'a mut [T],
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: isize,
}

fn write<'a, T: Elem>(w: &'a mut TensorWrite<'_>) -> Result<Option<Out<'a, T>>> {
    Ok(match w {
        TensorWrite::Tensor(t) => match t.as_typed_mut::<T>() {
            Some(t) => {
                let shape = t.shape().to_vec();
                let strides = col_major_strides(&shape)?;
                Some(Out {
                    data: t.host_data_mut()?,
                    shape,
                    strides,
                    offset: 0,
                })
            }
            None => None,
        },
        TensorWrite::View(v) => match T::view_mut(v) {
            Some(v) => {
                let (shape, strides, offset) =
                    (v.shape().to_vec(), v.strides().to_vec(), v.offset());
                Some(Out {
                    data: v.host_storage_mut()?,
                    shape,
                    strides,
                    offset,
                })
            }
            None => None,
        },
    })
}

/// The operand's layout with its conjugation. Conjugating a real operand is
/// the identity, so it is not requested.
fn spec<T: Elem>(
    dims: &[usize],
    strides: &[isize],
    offset: isize,
    conj: bool,
) -> std::result::Result<OperandSpec, tprims_contract::Error> {
    let op = if conj && T::STORAGE.is_complex() {
        Op::Conjugate
    } else {
        Op::Identity
    };
    Ok(OperandSpec::new(LayoutSpec::new(dims, strides, offset)?).with_op(op))
}

/// The plan of one `dot_general` over the given operand layouts, `D` being
/// accumulated in place (`C` is `D`).
fn plan_dot<T: Elem>(
    dot: &DotGeneral,
    a: OperandSpec,
    b: OperandSpec,
    d: OperandSpec,
) -> std::result::Result<Plan<T>, tprims_contract::Error> {
    let problem = Problem::from_dot_general(T::STORAGE, a, b, d, dot)?;
    Plan::new(&problem, &PlanConfig::default())
}

fn failure(op: &'static str, e: impl std::fmt::Display) -> Error {
    Error::backend_failure(op, format!("tprims: {e}"))
}

const UNSUPPORTED_LAYOUT_OUT: CpuProviderOutcome =
    CpuProviderOutcome::Unsupported(CpuProviderUnsupported::Layout(CpuOperand::Output));

/// dtype dispatch for the four supported element types.
macro_rules! by_dtype {
    ($dtype:expr, $f:ident($($arg:expr),*)) => {
        match $dtype {
            DType::F32 => $f::<f32>($($arg),*),
            DType::F64 => $f::<f64>($($arg),*),
            DType::C32 => $f::<Complex32>($($arg),*),
            DType::C64 => $f::<Complex64>($($arg),*),
            other => Ok(CpuProviderOutcome::Unsupported(CpuProviderUnsupported::DType(other))),
        }
    };
}

/// A `[rows, cols]` or `[rows, cols, batch]` view of `layout` in `data`.
fn mat_dims(
    rows: usize,
    cols: usize,
    batch: Option<usize>,
    l: CpuBatchedMatrixLayout,
) -> (Vec<usize>, Vec<isize>) {
    match batch {
        None => (vec![rows, cols], vec![l.row_stride(), l.column_stride()]),
        Some(b) => (
            vec![rows, cols, b],
            vec![l.row_stride(), l.column_stride(), l.batch_stride()],
        ),
    }
}

fn gemm_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    mut request: CpuGemmRequest<'_, '_, '_>,
    batched: bool,
) -> Result<CpuProviderOutcome> {
    const OP: &str = "gemm";
    if request.vendor_batch() == CpuVendorBatch::Required {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::RuntimeUnavailable,
        ));
    }
    let acc = request.accumulation();
    let (Some(alpha), Some(beta)) = (T::scalar(acc.alpha), T::scalar(acc.beta)) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Accumulation,
        ));
    };
    let (m, n, k, count) = (
        request.rows(),
        request.columns(),
        request.contracted(),
        request.batch_count(),
    );
    let (la, lb, lc) = (
        request.lhs_layout(),
        request.rhs_layout(),
        request.output_layout(),
    );
    let (lhs, rhs) = (request.lhs().clone(), request.rhs().clone());
    let (Some(a), Some(b)) = (read::<T>(&lhs)?, read::<T>(&rhs)?) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ));
    };
    // A batch is a batch axis of one contraction: `[m, k, b] x [k, n, b]`.
    let batch = (batched || count != 1).then_some(count);
    let (da, sa) = mat_dims(m, k, batch, la);
    let (db, sb) = mat_dims(k, n, batch, lb);
    let (dc, sc) = mat_dims(m, n, batch, lc);
    let av = StridedView::new(a.data, &da, &sa, la.offset()).map_err(|e| failure(OP, e))?;
    let bv = StridedView::new(b.data, &db, &sb, lb.offset()).map_err(|e| failure(OP, e))?;
    let output = request.output();
    let Some(c) = write::<T>(output)? else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    let mut cv = StridedViewMut::new(c.data, &dc, &sc, lc.offset()).map_err(|e| failure(OP, e))?;
    let dot = match batch {
        None => DotGeneral::new(&[1], &[0], &[], &[]),
        Some(_) => DotGeneral::new(&[1], &[0], &[2], &[2]),
    };
    let plan = spec::<T>(&da, &sa, la.offset(), acc.lhs_conj)
        .and_then(|a| {
            let b = spec::<T>(&db, &sb, lb.offset(), acc.rhs_conj)?;
            let d = spec::<T>(&dc, &sc, lc.offset(), false)?;
            plan_dot::<T>(&dot, a, b, d)
        })
        .map_err(|e| failure(OP, e))?;
    // The plan validates the views before writing, so an error leaves the
    // output intact.
    with_exec(ctx, |exec| {
        plan.execute_into_accum(
            exec,
            alpha,
            &av,
            &bv,
            beta,
            AccumulationSource::Output,
            &mut cv,
        )
    })
    .map_err(|e| failure(OP, e))?;
    Ok(CpuProviderOutcome::Executed)
}

/// Independent jobs of different sizes over shared buffers: one plan per
/// distinct `(rows, inner, cols)`, all jobs run as one batch (each item
/// carries its own plan), so many small jobs spread over the pool.
fn grouped_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    mut request: CpuGroupedGemmRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    const OP: &str = "grouped_gemm";
    let acc = request.accumulation();
    let (Some(alpha), Some(beta)) = (T::scalar(acc.alpha), T::scalar(acc.beta)) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Accumulation,
        ));
    };
    let (lhs, rhs) = (request.lhs().clone(), request.rhs().clone());
    let (Some(a), Some(b)) = (read::<T>(&lhs)?, read::<T>(&rhs)?) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ));
    };
    // Job offsets are relative to each operand's view offset.
    let base = |o: isize| usize::try_from(o).map_err(|_| failure(OP, "negative operand offset"));
    let (oa, ob) = (base(a.offset)?, base(b.offset)?);
    // A job with no output elements touches nothing.
    let mut jobs: Vec<_> = request
        .jobs()
        .iter()
        .filter(|j| j.rows() > 0 && j.cols() > 0)
        .copied()
        .collect();
    let output = request.output();
    let Some(c) = write::<T>(output)? else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    let oc = base(c.offset)?;
    // Output blocks are compact column-major; hand each job its own disjoint
    // sub-slice of the output, in offset order.
    jobs.sort_by_key(|j| j.out_offset());
    let (cs, ca) = (acc.lhs_conj, acc.rhs_conj);
    let mut plans: HashMap<(usize, usize, usize), Plan<T>> = HashMap::new();
    for j in &jobs {
        let key = (j.rows(), j.contracted(), j.cols());
        if plans.contains_key(&key) {
            continue;
        }
        let (rows, inner, cols) = key;
        let col = |r: usize, c: usize| (vec![r, c], vec![1, r.max(1) as isize]);
        let ((da, sa), (db, sb), (dc, sc)) = (col(rows, inner), col(inner, cols), col(rows, cols));
        let plan = spec::<T>(&da, &sa, 0, cs)
            .and_then(|a| {
                let b = spec::<T>(&db, &sb, 0, ca)?;
                let d = spec::<T>(&dc, &sc, 0, false)?;
                plan_dot::<T>(&DotGeneral::new(&[1], &[0], &[], &[]), a, b, d)
            })
            .map_err(|e| failure(OP, e))?;
        plans.insert(key, plan);
    }
    let mut rest = c.data;
    let mut pos = 0usize;
    let mut items = Vec::with_capacity(jobs.len());
    for j in &jobs {
        let (rows, inner, cols) = (j.rows(), j.contracted(), j.cols());
        let start = oc + j.out_offset();
        let len = rows * cols;
        if start < pos {
            return Err(failure(OP, "overlapping output blocks"));
        }
        let tail = std::mem::take(&mut rest);
        if start - pos > tail.len() || len > tail.len() - (start - pos) {
            return Err(failure(OP, "output block exceeds the buffer"));
        }
        let (_, tail) = tail.split_at_mut(start - pos);
        let (block, tail) = tail.split_at_mut(len);
        rest = tail;
        pos = start + len;
        let (sa, sb, sc) = (
            [1, rows.max(1) as isize],
            [1, inner.max(1) as isize],
            [1, rows.max(1) as isize],
        );
        let av = StridedView::new(a.data, &[rows, inner], &sa, (oa + j.lhs_offset()) as isize)
            .map_err(|e| failure(OP, e))?;
        let bv = StridedView::new(b.data, &[inner, cols], &sb, (ob + j.rhs_offset()) as isize)
            .map_err(|e| failure(OP, e))?;
        let dv = StridedViewMut::new(block, &[rows, cols], &sc, 0).map_err(|e| failure(OP, e))?;
        items.push(BatchItem {
            plan: &plans[&(rows, inner, cols)],
            alpha,
            a: av,
            b: bv,
            beta,
            source: Some(AccumulationSource::Output),
            d: dv,
        });
    }
    // The batch validates every item before writing any.
    with_exec(ctx, |exec| contract_batched(&mut items, exec)).map_err(|e| failure(OP, e))?;
    Ok(CpuProviderOutcome::Executed)
}

fn dot_general_typed<T: Elem>(
    ctx: &CpuExecutionContext<'_>,
    request: CpuDotGeneralRequest<'_, '_, '_>,
) -> Result<CpuProviderOutcome> {
    const OP: &str = "dot_general";
    let (lhs, rhs, output, axes, acc) = request.into_parts();
    let (Some(alpha), Some(beta)) = (T::scalar(acc.alpha), T::scalar(acc.beta)) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::Accumulation,
        ));
    };
    let (Some(a), Some(b)) = (read::<T>(lhs)?, read::<T>(rhs)?) else {
        return Ok(CpuProviderOutcome::Unsupported(
            CpuProviderUnsupported::DType(lhs.dtype()),
        ));
    };
    let (lc, rc): (Vec<usize>, Vec<usize>) = axes.contracting_pairs().unzip();
    let (lb, rb): (Vec<usize>, Vec<usize>) = axes.batch_pairs().unzip();
    let dot = DotGeneral::new(&lc, &rc, &lb, &rb);
    let Some(c) = write::<T>(output)? else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    // Output order [lhs free, rhs free, batch] is the same in both libraries.
    // A plan the library cannot build is unsupported, not an error: nothing
    // has been written yet.
    let Ok(plan) = spec::<T>(&a.shape, &a.strides, a.offset, acc.lhs_conj).and_then(|sa| {
        let sb = spec::<T>(&b.shape, &b.strides, b.offset, acc.rhs_conj)?;
        let sd = spec::<T>(&c.shape, &c.strides, c.offset, false)?;
        plan_dot::<T>(&dot, sa, sb, sd)
    }) else {
        return Ok(UNSUPPORTED_LAYOUT_OUT);
    };
    let av =
        StridedView::new(a.data, &a.shape, &a.strides, a.offset).map_err(|e| failure(OP, e))?;
    let bv =
        StridedView::new(b.data, &b.shape, &b.strides, b.offset).map_err(|e| failure(OP, e))?;
    let mut cv =
        StridedViewMut::new(c.data, &c.shape, &c.strides, c.offset).map_err(|e| failure(OP, e))?;
    with_exec(ctx, |exec| {
        plan.execute_into_accum(
            exec,
            alpha,
            &av,
            &bv,
            beta,
            AccumulationSource::Output,
            &mut cv,
        )
    })
    .map_err(|e| failure(OP, e))?;
    Ok(CpuProviderOutcome::Executed)
}

impl CpuGemmProvider for TprimsProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        capabilities()
    }

    /// One GEMM through a `tprims_contract::Plan`.
    ///
    /// # Errors
    ///
    /// [`Error::BackendFailure`] when tprims rejects the validated request
    /// (output untouched) or a host buffer is unavailable.
    fn gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), gemm_typed(context, request, false))
    }

    /// A strided batch through a `tprims_contract::Plan` with a batch axis.
    ///
    /// # Errors
    ///
    /// As [`CpuGemmProvider::gemm`].
    fn strided_batched_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGemmRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), gemm_typed(context, request, true))
    }

    /// Variable-size jobs through `tprims_contract::contract_batched`.
    ///
    /// # Errors
    ///
    /// As [`CpuGemmProvider::gemm`].
    fn grouped_gemm(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuGroupedGemmRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), grouped_typed(context, request))
    }
}

impl CpuGeneralContractionProvider for TprimsProvider {
    fn execution_capabilities(&self) -> CpuProviderExecutionCapabilities {
        capabilities()
    }

    /// A binary contraction through a `tprims_contract::Plan`.
    ///
    /// # Errors
    ///
    /// As [`CpuGemmProvider::gemm`]; layouts tprims cannot plan are reported
    /// as unsupported instead.
    fn dot_general(
        &self,
        context: &CpuExecutionContext<'_>,
        request: CpuDotGeneralRequest<'_, '_, '_>,
    ) -> Result<CpuProviderOutcome> {
        by_dtype!(request.lhs().dtype(), dot_general_typed(context, request))
    }
}
