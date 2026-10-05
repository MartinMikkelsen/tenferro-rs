//! Adapter from tenferro's CPU session to the extracted LAPACK/BLAS provider (`tlinalg-blas`).
//!
//! The host keeps policy, exactly as on the faer route: the batch strategy is admitted here, and
//! the provider is called once for the whole batch. Unlike the faer adapter there are no lanes and
//! no parallelism token: the vendor owns threading inside every call, so tenferro never fans out
//! around it and `tlinalg-blas` loops over the batch serially. Pooled scratch reaches the provider
//! through [`super::tlinalg_workspace::TlinalgWorkspace`].

#![cfg(feature = "cpu-blas")]

use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_cpu::CpuExecutionContext;
use tlinalg_blas::lu::{lu_factor, lu_factor_solve, lu_solve_prepared};
use tlinalg_blas::{LapackScalar, Op};

use super::linalg::blas::check_packed_lu_batch_strategy;
use super::tlinalg_error::map_blas_error;
use super::tlinalg_workspace::TlinalgWorkspace;

/// The session pool, as the provider's scratch contract.
pub(crate) fn workspace(pool: &mut BufferPool) -> TlinalgWorkspace<'_> {
    TlinalgWorkspace::new(pool)
}

/// Factor `batch` compact `m x n` matrices in place.
pub(crate) fn factor_batch<T: LapackScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> tenferro_tensor::Result<()> {
    check_packed_lu_batch_strategy(ctx, op.as_str(), parity.len())?;
    lu_factor(op, m, n, lu, pivots, parity).map_err(|error| map_blas_error(op, error))
}

/// Solve `op(A) X = B` for `batch` compact systems from packed factors.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared_batch<T: LapackScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &[T],
    pivots: &[i32],
    output: &mut [T],
    transpose_a: bool,
    conjugate_a: bool,
) -> tenferro_tensor::Result<()> {
    let batch = pivots.len().checked_div(n).unwrap_or(0);
    check_packed_lu_batch_strategy(ctx, op.as_str(), batch)?;
    lu_solve_prepared(
        op,
        n,
        nrhs,
        packed_lu,
        pivots,
        output,
        transpose_a,
        conjugate_a,
    )
    .map_err(|error| map_blas_error(op, error))
}

/// Factor and solve `A X = B` for `batch` compact systems, keeping the packed factors.
pub(crate) fn factor_solve_batch<T: LapackScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    let batch = pivots.len().checked_div(n).unwrap_or(0);
    check_packed_lu_batch_strategy(ctx, op.as_str(), batch)?;
    lu_factor_solve(op, n, nrhs, packed_lu, pivots, output)
        .map_err(|error| map_blas_error(op, error))
}
