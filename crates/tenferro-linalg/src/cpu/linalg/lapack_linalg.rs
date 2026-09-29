pub(crate) use crate::error::unsupported_dtype;

mod cholesky;
mod eig;
mod eigh;
mod full_piv_lu;
mod helpers;
mod lu;
mod qr;
mod solve;
mod svd;
mod triangular_solve;

pub(crate) use cholesky::{cholesky, cholesky_compact_data};
pub(crate) use eig::{eig, eig_values};
pub(crate) use eigh::{eigh, eigh_values};
pub(crate) use full_piv_lu::{full_piv_lu, full_piv_lu_solve};
pub(crate) use lu::{lu, lu_factor};
pub(crate) use qr::{
    append_2d as householder_qr_append, compact_factor_2d as householder_qr,
    from_factors_2d as householder_qr_from_factors, q_columns_2d as householder_qr_q_columns, qr,
    rank_revealing_qr, raw_r_2d as householder_qr_r,
};
pub(crate) use solve::{
    lu_factor_solve_batched_in_place, lu_solve_prepared_batched_in_place, solve, solve_from_views,
    solve_into,
};
pub(crate) use svd::{svd, svd_full, svd_values};
pub(crate) use triangular_solve::triangular_solve;

/// Reject a forced batch strategy the LAPACK packed-LU loop cannot honor.
///
/// The loop calls one provider-threaded `?getrf`/`?getrs` per matrix, so only
/// `Auto` and `ProviderItems` (sequential batch, provider-managed items)
/// describe it. `Sequential` cannot stop the provider's own threads,
/// `OuterParallel` would put several provider-threaded calls in flight at
/// once, and there is no vendor batched factorization binding. A single matrix
/// is not a batch, so the policy does not apply to it.
pub(crate) fn check_packed_lu_batch_strategy(
    ctx: &tenferro_cpu::CpuExecutionContext<'_>,
    op: &'static str,
    batch: usize,
) -> tenferro_tensor::Result<()> {
    use tenferro_cpu::CpuBatchStrategy;

    let strategy = ctx.batch_policy().strategy();
    if batch <= 1
        || matches!(
            strategy,
            CpuBatchStrategy::Auto | CpuBatchStrategy::ProviderItems
        )
    {
        return Ok(());
    }
    let reason = match strategy {
        CpuBatchStrategy::Sequential => "the BLAS provider threads each call itself",
        CpuBatchStrategy::OuterParallel => {
            "provider-threaded LAPACK calls cannot run concurrently in tenferro lanes"
        }
        _ => "no vendor batched factorization binding exists",
    };
    Err(tenferro_tensor::Error::unsupported(
        op,
        format!(
            "batch strategy {strategy:?} is not available: {reason}; use CpuBatchStrategy::Auto or ProviderItems"
        ),
    ))
}
