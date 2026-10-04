use num_complex::{Complex32, Complex64};
use tlinalg_traits::IndexWorkspace;

use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;

use super::helpers::{
    batch_element_count, checked_product, has_zero_dim, matrix_with_batch_shape, pooled_copy,
    pooled_zeroed, release_scratch, split_core_and_batch_result, tensor_from_vec_with_template,
    vector_with_batch_shape,
};

/// Which singular factors a batched SVD computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SvdMode {
    /// Thin factors: `U` is `m x k` and `Vt` is `k x n`.
    Thin,
    /// Full factors: `U` is `m x m` and `Vt` is `n x n`, so the trailing
    /// columns and rows span the left and right nullspaces.
    Full,
    /// Singular values only.
    Values,
}

impl SvdMode {
    /// `(U columns, Vt rows)` of one factor pair; zero in values-only mode.
    fn factor_dims(self, m: usize, n: usize) -> (usize, usize) {
        let k = m.min(n);
        match self {
            Self::Thin => (k, k),
            Self::Full => (m, n),
            Self::Values => (0, 0),
        }
    }
}

pub(crate) trait LapackSvd:
    Clone + Copy + Default + PoolScalar + tlinalg_blas::LapackScalar
{
    /// The multiplicative unit, used to build the identity factor a full
    /// decomposition still owes for an empty core dimension.
    fn unit() -> Self;

    /// Real singular values in this scalar type, reusing the buffer when the
    /// scalar is real.
    fn values_as_scalar(
        buffers: &mut BufferPool,
        values: Vec<<Self as tlinalg_blas::symbols::Symbols>::Real>,
    ) -> Vec<Self>;
}

/// The real scalar of an SVD scalar.
pub(crate) type SvdRealOf<T> = <T as tlinalg_blas::symbols::Symbols>::Real;

/// The four pooled buffers one batched SVD fills: the destroyed input, the values, and the two
/// factors.
type SvdBuffers<T> = (Vec<T>, Vec<SvdRealOf<T>>, Vec<T>, Vec<T>);

/// The real scalar of an SVD scalar, with every bound the outputs and the pooled scratch need.
pub(crate) trait SvdReal:
    Clone + Copy + Default + PoolScalar + tenferro_tensor::TensorScalar + tlinalg_traits::Scalar
{
}

impl SvdReal for f32 {}
impl SvdReal for f64 {}

/// The scratch an SVD call needs, in the shape the extracted kernel asks for.
///
/// The kernel wants a workspace over the scalar, a workspace over its real scalar and an integer
/// workspace. Naming them as associated types lets the generic wrappers below prove those bounds
/// once instead of threading three where-clauses through every function that reaches the kernel.
pub(crate) trait SvdScratch: LapackSvd {
    /// The workspace type for one call.
    type Workspace<'a>: tlinalg_traits::Workspace<Self>
        + tlinalg_traits::Workspace<<Self as tlinalg_blas::symbols::Symbols>::Real>
        + IndexWorkspace;

    /// Wrap the session's pool for one call.
    fn workspace(pool: &mut BufferPool) -> Self::Workspace<'_>;
}

macro_rules! impl_svd_scratch {
    ($scalar:ty) => {
        impl SvdScratch for $scalar {
            type Workspace<'a> = crate::cpu::tlinalg_workspace::TlinalgWorkspace<'a>;

            fn workspace(pool: &mut BufferPool) -> Self::Workspace<'_> {
                crate::cpu::tlinalg_workspace::TlinalgWorkspace::new(pool)
            }
        }
    };
}

impl_svd_scratch!(f32);
impl_svd_scratch!(f64);
impl_svd_scratch!(Complex32);
impl_svd_scratch!(Complex64);

macro_rules! impl_real_svd {
    ($scalar:ty) => {
        impl LapackSvd for $scalar {
            fn unit() -> Self {
                1.0
            }

            fn values_as_scalar(_buffers: &mut BufferPool, values: Vec<Self>) -> Vec<Self> {
                values
            }
        }
    };
}

macro_rules! impl_complex_svd {
    ($complex:ty, $real:ty) => {
        impl LapackSvd for $complex {
            fn unit() -> Self {
                <$complex>::new(1.0, 0.0)
            }

            fn values_as_scalar(buffers: &mut BufferPool, values: Vec<$real>) -> Vec<Self> {
                let mut converted = buffers.acquire_with_capacity::<$complex>(values.len());
                converted.extend(values.iter().map(|&value| <$complex>::new(value, 0.0)));
                release_scratch(buffers, values);
                converted
            }
        }
    };
}

impl_real_svd!(f32);
impl_real_svd!(f64);
impl_complex_svd!(Complex32, f32);
impl_complex_svd!(Complex64, f64);

/// Pooled factor buffers for a batched SVD in `mode`, returned as
/// `(a copy, values, U, Vt)`.
#[allow(clippy::type_complexity)]
/// The extracted kernel's spelling of a mode.
fn tlinalg_svd_mode(mode: SvdMode) -> tlinalg_blas::svd::SvdMode {
    match mode {
        SvdMode::Thin => tlinalg_blas::svd::SvdMode::Thin,
        SvdMode::Full => tlinalg_blas::svd::SvdMode::Full,
        SvdMode::Values => tlinalg_blas::svd::SvdMode::Values,
    }
}

fn svd_buffers<T: SvdScratch>(
    buffers: &mut BufferPool,
    op: &'static str,
    mode: SvdMode,
    m: usize,
    n: usize,
    batch_shape: &[usize],
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<SvdBuffers<T>>
where
    <T as tlinalg_blas::symbols::Symbols>::Real: SvdReal,
{
    let batch = batch_element_count(op, batch_shape)?;
    let (u_cols, vt_rows) = mode.factor_dims(m, n);
    let s_len = checked_product(op, "singular values", &[m.min(n), batch])?;
    let u_len = checked_product(op, "left singular vectors", &[m, u_cols, batch])?;
    let vt_len = checked_product(op, "right singular vectors", &[vt_rows, n, batch])?;
    let mut a = pooled_copy(buffers, input.host_data()?);
    let mut s = pooled_zeroed::<SvdRealOf<T>>(buffers, s_len);
    let mut u = pooled_zeroed::<T>(buffers, u_len);
    let mut vt = pooled_zeroed::<T>(buffers, vt_len);
    tlinalg_blas::svd::svd_batch(
        tlinalg_traits::Op::Svd,
        tlinalg_svd_mode(mode),
        m,
        n,
        &mut a,
        &mut s,
        &mut u,
        &mut vt,
        &mut T::workspace(buffers),
        tlinalg_traits::Parallel::Sequential,
    )
    .map_err(|error| crate::cpu::tlinalg_error::map_error(tlinalg_traits::Op::Svd, error))?;
    Ok((a, s, u, vt))
}

pub(crate) fn svd<T: SvdScratch>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>
where
    <T as tlinalg_blas::symbols::Symbols>::Real: SvdReal,
{
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(
                matrix_with_batch_shape(m, k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                matrix_with_batch_shape(k, n, batch_shape),
                Vec::new(),
                input,
            )?,
        ]);
    }
    let (a, s, u, vt) = svd_buffers(buffers, "svd", SvdMode::Thin, m, n, batch_shape, input)?;
    release_scratch(buffers, a);
    let s = T::values_as_scalar(buffers, s);
    Ok(vec![
        tensor_from_vec_with_template(matrix_with_batch_shape(m, k, batch_shape), u, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), s, input)?,
        tensor_from_vec_with_template(matrix_with_batch_shape(k, n, batch_shape), vt, input)?,
    ])
}

pub(crate) fn svd_full<T: SvdScratch>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>
where
    <T as tlinalg_blas::symbols::Symbols>::Real: SvdReal,
{
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_full")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    if has_zero_dim(input.shape()) {
        return empty_full_svd_outputs("svd_full", m, n, batch_shape, input);
    }
    let (a, s, u, vt) = svd_buffers(buffers, "svd_full", SvdMode::Full, m, n, batch_shape, input)?;
    release_scratch(buffers, a);
    let s = T::values_as_scalar(buffers, s);
    Ok(vec![
        tensor_from_vec_with_template(matrix_with_batch_shape(m, m, batch_shape), u, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(m.min(n), batch_shape), s, input)?,
        tensor_from_vec_with_template(matrix_with_batch_shape(n, n, batch_shape), vt, input)?,
    ])
}

/// Full-SVD factors for an input with an empty core dimension.
///
/// The full variant keeps its `m x m` and `n x n` output shapes even when the
/// other core dimension is zero, so the factor for the non-empty dimension is
/// the identity rather than an empty tensor. This mirrors the faer provider so
/// one public call has one shape and unitarity contract.
fn empty_full_svd_outputs<T: SvdScratch, U>(
    op: &'static str,
    m: usize,
    n: usize,
    batch_shape: &[usize],
    template: &TypedTensor<U>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>>
where
    <T as tlinalg_blas::symbols::Symbols>::Real: SvdReal,
{
    let blocks = checked_product(op, "batch shape", batch_shape)?;
    Ok(vec![
        tensor_from_vec_with_template(
            matrix_with_batch_shape(m, m, batch_shape),
            identity_blocks::<T>(op, m, blocks)?,
            template,
        )?,
        tensor_from_vec_with_template(
            vector_with_batch_shape(m.min(n), batch_shape),
            Vec::new(),
            template,
        )?,
        tensor_from_vec_with_template(
            matrix_with_batch_shape(n, n, batch_shape),
            identity_blocks::<T>(op, n, blocks)?,
            template,
        )?,
    ])
}

/// `blocks` column-major `dim x dim` identity matrices laid out back to back.
fn identity_blocks<T: SvdScratch>(
    op: &'static str,
    dim: usize,
    blocks: usize,
) -> tenferro_tensor::Result<Vec<T>>
where
    <T as tlinalg_blas::symbols::Symbols>::Real: SvdReal,
{
    let per_block = checked_product(op, "identity block", &[dim, dim])?;
    let len = checked_product(op, "identity stack", &[per_block, blocks])?;
    let mut data = vec![T::default(); len];
    for block in 0..blocks {
        let base = block * per_block;
        for index in 0..dim {
            data[base + index + index * dim] = T::unit();
        }
    }
    Ok(data)
}

pub(crate) fn svd_values<T: SvdScratch>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<TypedTensor<SvdRealOf<T>>>
where
    <T as tlinalg_blas::symbols::Symbols>::Real: SvdReal,
{
    let (matrix_shape, batch_shape) = split_core_and_batch_result(input, 2, "svd_values")?;
    let (m, n) = (matrix_shape[0], matrix_shape[1]);
    let k = m.min(n);
    if has_zero_dim(input.shape()) {
        return tensor_from_vec_with_template(
            vector_with_batch_shape(k, batch_shape),
            Vec::new(),
            input,
        );
    }
    let (a, s, u, vt) = svd_buffers(
        buffers,
        "svd_values",
        SvdMode::Values,
        m,
        n,
        batch_shape,
        input,
    )?;
    release_scratch(buffers, a);
    release_scratch(buffers, u);
    release_scratch(buffers, vt);
    tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), s, input)
}
