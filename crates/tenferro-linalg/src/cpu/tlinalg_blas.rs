//! Adapter from tenferro's CPU session to the extracted LAPACK/BLAS implementation.
//!
//! The host keeps policy, exactly as on the faer route: the batch strategy is admitted here, and the
//! kernel is called once for the whole batch. Two things differ from the faer adapter on purpose:
//!
//! - **No lanes.** The vendor owns threading inside a batch call, so tenferro never fans out around
//!   it. `check_packed_lu_batch_strategy` is the existing admission rule that rejects the strategies
//!   a provider-threaded implementation cannot serve.
//! - **The token is ignored.** `tlinalg-blas` accepts a `Parallel` for interface parity and does not
//!   use it, so this adapter passes the sequential token rather than pretending the host controls
//!   vendor workers.

#![cfg(feature = "cpu-blas")]

use tenferro_cpu::CpuExecutionContext;
use tlinalg_blas::lu::{factor_chunk, factor_solve_chunk, solve_prepared_chunk};
use tlinalg_blas::LapackScalar;
use tlinalg_traits::{Op, Parallel};

use super::linalg::blas::check_packed_lu_batch_strategy;
use super::tlinalg_error::map_error;

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
    factor_chunk(op, m, n, lu, pivots, parity, Parallel::Sequential)
        .map_err(|error| map_error(op, error))
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
    solve_prepared_chunk(
        op,
        n,
        nrhs,
        packed_lu,
        pivots,
        output,
        transpose_a,
        conjugate_a,
        Parallel::Sequential,
    )
    .map_err(|error| map_error(op, error))
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
    factor_solve_chunk(op, n, nrhs, packed_lu, pivots, output, Parallel::Sequential)
        .map_err(|error| map_error(op, error))
}
