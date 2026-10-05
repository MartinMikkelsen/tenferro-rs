//! Adapter from tenferro's CPU session to the extracted `tlinalg` (faer) provider.
//!
//! The host keeps policy: this module turns the session's resolved batch strategy into a
//! [`LanePlan`], derives the [`Parallel`] token from the entered context, describes tensors to the
//! provider as borrowed strided operands, and owns the error mapping. `tlinalg` owns the kernels,
//! their scratch, the batch loop and the batch-direction lane fan-out on the pool it is handed.
//!
//! Every family reaches the provider through one batched call: the provider never sees a tensor,
//! and the host never loops over the batch itself.

#![cfg(feature = "cpu-faer")]

use strided_view::RawStridedRef;
use tenferro_cpu::{CpuBatchStrategy, CpuExecutionContext};
use tlinalg::packed_lu::{factor, factor_solve, solve_prepared, validate_pivots};
use tlinalg::{FaerScalar, LanePlan, Op, Parallel};

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
/// One policy for every batched faer family: forced `Sequential` and `ProviderItems` stay on one
/// lane, forced `OuterParallel` needs a context that can fan out, and a vendor-batched strategy is
/// unavailable because the implementation has no vendor batching. `Auto` fans out over the
/// context's budget once the thresholds say the batch is large enough. A single matrix is not a
/// batch, so the policy does not apply to it.
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
        CpuBatchStrategy::Sequential => Ok(LanePlan::sequential()),
        CpuBatchStrategy::ProviderItems => Ok(LanePlan::single(parallel)),
        _ => Err(unavailable(
            "the linalg provider has no vendor batched factorization",
        )),
    }
}

/// Largest per-item matrix dimension, `max(rows, cols)`, for which `Auto` may fan a batch of the
/// families other than packed LU out over lanes.
///
/// Provisional guard pending a work-model thread policy for CPU linalg
/// (tensor4all/tenferro-rs#2000). Before the batched extraction only the packed-LU family fanned
/// out; every other family ran its batch serially with the context's parallelism inside each item.
/// Lane fan-out pays off for many small matrices, where one item cannot use the budget, but for a
/// large matrix faer's own parallelism over one item is the better use of the threads, and the
/// item-count thresholds alone cannot tell the two apart. So above this size `Auto` keeps exactly
/// the pre-extraction behaviour. Forced strategies are unaffected.
pub(crate) const AUTO_FAN_OUT_MAX_ITEM_DIM: usize = 64;

/// [`lane_plan`] for a family whose `Auto` fan-out is limited to small items.
///
/// Under `Auto` an item with `max(rows, cols)` above [`AUTO_FAN_OUT_MAX_ITEM_DIM`] runs on one
/// lane with the context's parallelism; every other case, including every forced strategy, is
/// [`lane_plan`]'s. The packed-LU family keeps [`lane_plan`] itself.
pub(crate) fn lane_plan_for_item<'a>(
    ctx: &CpuExecutionContext<'a>,
    op: Op,
    batch: usize,
    max_item_dim: usize,
) -> tenferro_tensor::Result<LanePlan<'a>> {
    let plan = lane_plan(ctx, op, batch)?;
    let auto = batch <= 1 || ctx.batch_policy().strategy() == CpuBatchStrategy::Auto;
    if auto && max_item_dim > AUTO_FAN_OUT_MAX_ITEM_DIM {
        return Ok(LanePlan::single(parallel_from(ctx)));
    }
    Ok(plan)
}

/// The resolved `(Parallel, LanePlan)` pair one batched call of a non-packed-LU family takes;
/// `max_item_dim` is `max(rows, cols)` of one item (see [`lane_plan_for_item`]).
pub(crate) fn execution<'a>(
    ctx: &CpuExecutionContext<'a>,
    op: Op,
    batch: usize,
    max_item_dim: usize,
) -> tenferro_tensor::Result<(Parallel<'a>, LanePlan<'a>)> {
    Ok((
        parallel_from(ctx),
        lane_plan_for_item(ctx, op, batch, max_item_dim)?,
    ))
}

/// The `(Parallel, LanePlan)` pair of the packed-LU family, which keeps [`lane_plan`] unchanged.
fn packed_lu_execution<'a>(
    ctx: &CpuExecutionContext<'a>,
    op: Op,
    batch: usize,
) -> tenferro_tensor::Result<(Parallel<'a>, LanePlan<'a>)> {
    Ok((parallel_from(ctx), lane_plan(ctx, op, batch)?))
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
    let (par, plan) = packed_lu_execution(ctx, op, parity.len())?;
    factor(op, m, n, lu, pivots, parity, par, plan).map_err(|error| map_error(op, error))
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
    let batch = pivots.len().checked_div(n).unwrap_or(0);
    // The whole batch is validated before the provider runs, so an invalid pivot is reported
    // before any item mutates its output and before a batch-strategy error can mask it. A
    // zero-size solve returns before validation, as the previous implementation did.
    let has_rhs = n > 0 && nrhs > 0;
    if has_rhs {
        validate_pivots(Op::LuSolvePrepared, n, pivots)
            .map_err(|error| map_error(Op::LuSolvePrepared, error))?;
    }
    let (par, plan) = packed_lu_execution(ctx, op, batch)?;
    if !has_rhs {
        return Ok(());
    }
    let lu_dims = [n, n, batch];
    let lu_strides = compact_strides3(n, n);
    let pivot_dims = [n, batch];
    let pivot_strides = [1, n as isize];
    let packed_lu = RawStridedRef::new(packed_lu, &lu_dims, &lu_strides, 0)
        .map_err(|error| layout_error(op, error))?;
    let pivots = RawStridedRef::new(pivots, &pivot_dims, &pivot_strides, 0)
        .map_err(|error| layout_error(op, error))?;
    solve_prepared(
        op,
        packed_lu,
        pivots,
        nrhs,
        output,
        transpose_a,
        conjugate_a,
        par,
        plan,
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
    let batch = pivots.len().checked_div(n).unwrap_or(0);
    let (par, plan) = packed_lu_execution(ctx, op, batch)?;
    factor_solve(op, n, nrhs, packed_lu, pivots, output, par, plan)
        .map_err(|error| map_error(op, error))
}

/// Column-major strides of a compact `[rows, cols, batch]` stack.
fn compact_strides3(rows: usize, cols: usize) -> [isize; 3] {
    // INVARIANT: the caller's slice holds `rows * cols * batch` elements, so the per-matrix length
    // fits `usize`, and every in-bounds stride fits `isize` (Rust allocations never exceed it).
    [1, rows as isize, (rows * cols) as isize]
}

fn layout_error(op: Op, error: impl std::fmt::Display) -> tenferro_tensor::Error {
    tenferro_tensor::Error::invalid_argument(op.as_str(), "layout", error.to_string())
}
