//! Adapter from tenferro's CPU session to the extracted `tlinalg` crate.
//!
//! The host keeps policy: this module turns the session's resolved batch strategy into a lane
//! plan, drives the outer fan-out, and owns the error mapping. `tlinalg` owns the kernels and
//! their per-chunk scratch. Nothing here re-derives parallelism inside a lane — a lane child is
//! `ParallelMode::Sequential`, so `parallel_from` returns `Sequential` for it by construction.
//!
//! Nothing in the packed-LU family takes pooled buffers: the operands are the caller's slices and
//! the scratch is `tlinalg`'s own, so this adapter does not use `tlinalg_traits::Workspace`.

#![cfg(feature = "cpu-faer")]

use std::sync::Mutex;

use tenferro_cpu::{CpuBatchStrategy, CpuExecutionContext};
use tlinalg::packed_lu::{factor_chunk, factor_solve_chunk, solve_prepared_chunk, FactorScratch};
use tlinalg::FaerScalar;
use tlinalg_traits::{LanePlan, Op, Parallel};

use super::tlinalg_error::map_error;

/// The parallelism of one tenferro context, for `tlinalg`.
///
/// Derived from `rayon_pool()` alone, so the token agrees with the pool the session installed: a
/// sequential context, a one-thread budget, a non-Rayon executor, and a lane child all yield
/// [`Parallel::Sequential`], and a parallel context yields the pool the kernel must run on.
pub(crate) fn parallel_from<'a>(ctx: &CpuExecutionContext<'a>) -> Parallel<'a> {
    match ctx.rayon_pool() {
        Some(pool) => Parallel::Pool {
            pool,
            budget: ctx.thread_budget(),
        },
        None => Parallel::Sequential,
    }
}

/// Resolve the batch policy into a lane plan, rejecting strategies the implementation cannot serve.
///
/// Mirrors the policy the faer provider applied before the extraction: forced `Sequential` and
/// `ProviderItems` stay on one lane, forced `OuterParallel` needs a context that can fan out, and
/// a vendor-batched strategy is unavailable because the implementation has no vendor batching.
pub(crate) fn lane_plan<'a>(
    ctx: &CpuExecutionContext<'a>,
    op: Op,
    batch: usize,
) -> tenferro_tensor::Result<LanePlan<'a>> {
    let parallel = parallel_from(ctx);
    let budget_lanes = parallel.budget().get();
    let policy = ctx.batch_policy();
    let strategy = if batch > 1 {
        policy.strategy()
    } else {
        CpuBatchStrategy::Auto
    };
    let unavailable = |reason: &str| {
        tenferro_tensor::Error::unsupported(
            op.as_str(),
            format!(
                "batch strategy {strategy:?} is not available: {reason}; use CpuBatchStrategy::Auto or another strategy"
            ),
        )
    };
    match strategy {
        CpuBatchStrategy::Auto => Ok(LanePlan {
            lanes: if policy.thresholds().fans_out(batch, budget_lanes) {
                budget_lanes
            } else {
                1
            },
            item_parallel: parallel,
        }),
        CpuBatchStrategy::OuterParallel if ctx.can_fan_out_lanes() => Ok(LanePlan {
            lanes: budget_lanes.min(batch),
            item_parallel: Parallel::Sequential,
        }),
        CpuBatchStrategy::OuterParallel => Err(unavailable(
            "the context cannot fan out (one thread, or a sequential or nested context)",
        )),
        CpuBatchStrategy::Sequential => Ok(LanePlan {
            lanes: 1,
            item_parallel: Parallel::Sequential,
        }),
        CpuBatchStrategy::ProviderItems => Ok(LanePlan {
            lanes: 1,
            item_parallel: parallel,
        }),
        _ => Err(unavailable(
            "the linalg provider has no vendor batched factorization",
        )),
    }
}

/// Run `body` once per contiguous chunk, in the order the faer provider used.
///
/// `chunk_len` is `batch.div_ceil(lanes)`, so the last chunk can be shorter; the same whole-matrix
/// arithmetic applies to every buffer. The first failure wins and stops further work.
fn for_each_chunk<'a, I, F>(
    ctx: &CpuExecutionContext<'a>,
    jobs: I,
    body: F,
) -> tenferro_tensor::Result<()>
where
    I: IntoIterator,
    I::IntoIter: Send,
    I::Item: Send,
    F: Fn(I::Item, Parallel<'a>) -> tenferro_tensor::Result<()> + Sync,
{
    let failure = Mutex::new(None::<tenferro_tensor::Error>);
    let failed = |slot: &Mutex<Option<tenferro_tensor::Error>>| {
        slot.lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_some()
    };
    // Each lane is a sequential child of the context's own fan-out, so the kernel runs with the
    // sequential token and never nests a second fan-out.
    ctx.with_outer_lanes(jobs, |job, lane| {
        if failed(&failure) {
            return;
        }
        if let Err(error) = body(job, parallel_from(lane)) {
            let mut slot = failure
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if slot.is_none() {
                *slot = Some(error);
            }
        }
    });
    match failure
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
    {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// Factor `batch` compact `m x n` matrices in place, honouring the resolved lane plan.
pub(crate) fn factor_batch<T: FaerScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    m: usize,
    n: usize,
    lu: &mut [T],
    pivots: &mut [i32],
    parity: &mut [T],
) -> tenferro_tensor::Result<()> {
    let k = m.min(n);
    let matrix_len = m * n;
    let batch = parity.len();
    let plan = lane_plan(ctx, op, batch)?;
    if plan.lanes > 1 {
        let chunk_len = batch.div_ceil(plan.lanes);
        return for_each_chunk(
            ctx,
            lu.chunks_mut(chunk_len * matrix_len)
                .zip(pivots.chunks_mut(chunk_len * k))
                .zip(parity.chunks_mut(chunk_len)),
            |((lu_chunk, pivot_chunk), parity_chunk), par| {
                let mut scratch = FactorScratch::<T>::new(m, n, par);
                factor_chunk(
                    op,
                    m,
                    n,
                    lu_chunk,
                    pivot_chunk,
                    parity_chunk,
                    par,
                    &mut scratch,
                )
                .map_err(|error| map_error(op, error))
            },
        );
    }
    let mut scratch = FactorScratch::<T>::new(m, n, plan.item_parallel);
    factor_chunk(
        op,
        m,
        n,
        lu,
        pivots,
        parity,
        plan.item_parallel,
        &mut scratch,
    )
    .map_err(|error| map_error(op, error))
}

/// Solve `op(A) X = B` for `batch` compact systems from packed factors.
// INVARIANT: the argument list mirrors the tenferro session call it replaces; each value is a
// distinct operand or flag of the packed-LU solve contract, and grouping them would add a wrapper
// without removing an argument.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve_prepared_batch<T: FaerScalar>(
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
    let matrix_len = n * n;
    let rhs_len = n * nrhs;
    let batch = packed_lu.len() / matrix_len;
    // The whole batch is validated before the batch is split, so an invalid pivot is reported
    // before any chunk mutates its output and before a batch-strategy error can mask it. A
    // zero-size solve returns before validation, as the previous implementation did.
    if rhs_len > 0 {
        tlinalg::packed_lu::validate_pivots(Op::LuSolvePrepared, n, pivots)
            .map_err(|error| map_error(Op::LuSolvePrepared, error))?;
    }
    let plan = lane_plan(ctx, op, batch)?;
    if plan.lanes > 1 {
        let chunk_len = batch.div_ceil(plan.lanes);
        return for_each_chunk(
            ctx,
            packed_lu
                .chunks(chunk_len * matrix_len)
                .zip(pivots.chunks(chunk_len * n))
                .zip(output.chunks_mut(chunk_len * rhs_len)),
            |((matrix_chunk, ipiv_chunk), rhs_chunk), par| {
                solve_prepared_chunk(
                    op,
                    n,
                    nrhs,
                    matrix_chunk,
                    ipiv_chunk,
                    rhs_chunk,
                    transpose_a,
                    conjugate_a,
                    par,
                )
                .map_err(|error| map_error(op, error))
            },
        );
    }
    solve_prepared_chunk(
        op,
        n,
        nrhs,
        packed_lu,
        pivots,
        output,
        transpose_a,
        conjugate_a,
        plan.item_parallel,
    )
    .map_err(|error| map_error(op, error))
}

/// Factor and solve `A X = B` for `batch` compact systems, keeping the packed factors.
// INVARIANT: the argument list mirrors the tenferro session call it replaces; each value is a
// distinct operand or flag of the packed-LU solve contract, and grouping them would add a wrapper
// without removing an argument.
#[allow(clippy::too_many_arguments)]
pub(crate) fn factor_solve_batch<T: FaerScalar>(
    ctx: &CpuExecutionContext<'_>,
    op: Op,
    n: usize,
    nrhs: usize,
    packed_lu: &mut [T],
    pivots: &mut [i32],
    output: &mut [T],
) -> tenferro_tensor::Result<()> {
    let matrix_len = n * n;
    let rhs_len = n * nrhs;
    let batch = packed_lu.len() / matrix_len;
    let plan = lane_plan(ctx, op, batch)?;
    if plan.lanes > 1 {
        let chunk_len = batch.div_ceil(plan.lanes);
        // A zero-column RHS has no output chunk, and `chunks_mut(0)` is not a valid split, so the
        // zero-RHS case drives the lanes without an output buffer.
        if rhs_len == 0 {
            return for_each_chunk(
                ctx,
                packed_lu
                    .chunks_mut(chunk_len * matrix_len)
                    .zip(pivots.chunks_mut(chunk_len * n)),
                |(lu_chunk, pivot_chunk), par| {
                    let mut scratch = FactorScratch::<T>::new(n, n, par);
                    factor_solve_chunk(op, n, 0, lu_chunk, pivot_chunk, &mut [], par, &mut scratch)
                        .map_err(|error| map_error(op, error))
                },
            );
        }
        return for_each_chunk(
            ctx,
            packed_lu
                .chunks_mut(chunk_len * matrix_len)
                .zip(pivots.chunks_mut(chunk_len * n))
                .zip(output.chunks_mut(chunk_len * rhs_len)),
            |((lu_chunk, pivot_chunk), rhs_chunk), par| {
                let mut scratch = FactorScratch::<T>::new(n, n, par);
                factor_solve_chunk(
                    op,
                    n,
                    nrhs,
                    lu_chunk,
                    pivot_chunk,
                    rhs_chunk,
                    par,
                    &mut scratch,
                )
                .map_err(|error| map_error(op, error))
            },
        );
    }
    let mut scratch = FactorScratch::<T>::new(n, n, plan.item_parallel);
    factor_solve_chunk(
        op,
        n,
        nrhs,
        packed_lu,
        pivots,
        output,
        plan.item_parallel,
        &mut scratch,
    )
    .map_err(|error| map_error(op, error))
}
