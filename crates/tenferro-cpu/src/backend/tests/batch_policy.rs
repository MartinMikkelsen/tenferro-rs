//! Batch-policy precedence and scope restoration (#1938 D9).

use super::*;
use crate::{with_batch_policy, CpuBatchPolicy, CpuBatchStrategy};

fn observed(session: &mut dyn tenferro_tensor::BackendSession) -> CpuBatchStrategy {
    crate::with_cpu_exec_session(session, |cpu| {
        cpu.with_linalg_pool(|context, _| Ok(context.batch_policy().strategy()))
    })
    .expect("a CPU backend session")
    .unwrap()
}

#[test]
fn scoped_overrides_nest_over_the_backend_default_and_restore() {
    let mut backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Sequential));

    let seen = backend
        .with_backend_session(|session| {
            let mut seen = vec![observed(session)];
            with_batch_policy(
                session,
                CpuBatchPolicy::new(CpuBatchStrategy::ProviderItems),
                |session| {
                    seen.push(observed(session));
                    with_batch_policy(session, CpuBatchPolicy::default(), |session| {
                        seen.push(observed(session));
                    })
                    .unwrap();
                    seen.push(observed(session));
                },
            )
            .unwrap();
            seen.push(observed(session));
            seen
        })
        .unwrap();

    assert_eq!(
        seen,
        [
            CpuBatchStrategy::Sequential,
            CpuBatchStrategy::ProviderItems,
            CpuBatchStrategy::Auto,
            CpuBatchStrategy::ProviderItems,
            CpuBatchStrategy::Sequential,
        ]
    );
}

#[test]
fn scoped_override_is_restored_after_an_error_and_after_unwind() {
    let mut backend = CpuBackend::with_threads(1).unwrap();

    backend
        .with_backend_session(|session| {
            let returned = with_batch_policy(
                session,
                CpuBatchPolicy::new(CpuBatchStrategy::Sequential),
                |_| -> Result<(), &'static str> { Err("callback error") },
            )
            .unwrap();
            assert_eq!(returned, Err("callback error"));
            assert_eq!(observed(session), CpuBatchStrategy::Auto);

            let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = with_batch_policy(
                    session,
                    CpuBatchPolicy::new(CpuBatchStrategy::OuterParallel),
                    |_| panic!("forced scope panic"),
                );
            }));
            assert!(unwound.is_err());
            assert_eq!(observed(session), CpuBatchStrategy::Auto);
        })
        .unwrap();
}

#[test]
fn backend_default_policy_follows_clones_and_placements() {
    let backend = CpuBackend::with_threads(1)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::WholeBatchVendor));
    let mut clone = backend.clone();
    assert_eq!(
        clone.batch_policy().strategy(),
        CpuBatchStrategy::WholeBatchVendor
    );
    let strategy = clone.with_backend_session(observed).unwrap();
    assert_eq!(strategy, CpuBatchStrategy::WholeBatchVendor);
}

/// Batched contractions reach the vendor `cblas_?gemm_batch` binding (by
/// default for small items, or when forced) and agree with the per-item loop.
#[cfg(all(
    feature = "cpu-faer",
    any(feature = "blas-openblas", feature = "blas-mkl")
))]
#[test]
fn strided_batched_blas_vendor_route_matches_the_per_item_loop() {
    use tenferro_tensor::{DotGeneralConfig, Tensor, TensorRead};

    // [3, 4, 5] x [4, 2, 5] contracted over axis 1 / 0 with batch axis 2.
    let lhs_data: Vec<f64> = (0..60).map(|i| (i % 7) as f64 - 3.0).collect();
    let rhs_data: Vec<f64> = (0..40).map(|i| (i % 5) as f64 * 0.5).collect();
    let lhs = Tensor::from_vec_col_major(vec![3, 4, 5], lhs_data).unwrap();
    let rhs = Tensor::from_vec_col_major(vec![4, 2, 5], rhs_data).unwrap();
    let config = DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    };
    let run = |backend: &mut CpuBackend| {
        backend
            .with_backend_session(|session| {
                session.dot_general_read(
                    TensorRead::from_tensor(&lhs),
                    TensorRead::from_tensor(&rhs),
                    &config,
                )
            })
            .unwrap()
    };
    let expected =
        run(&mut CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap()).unwrap();
    for strategy in [
        CpuBatchStrategy::Auto,
        CpuBatchStrategy::WholeBatchVendor,
        CpuBatchStrategy::ProviderItems,
    ] {
        let mut backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas)
            .unwrap()
            .with_batch_policy(CpuBatchPolicy::new(strategy));
        let actual = run(&mut backend).unwrap();
        assert_eq!(actual.shape(), expected.shape());
        for (a, e) in actual
            .as_slice::<f64>()
            .unwrap()
            .iter()
            .zip(expected.as_slice::<f64>().unwrap())
        {
            assert!((a - e).abs() < 1e-12, "{strategy:?}: {a} vs {e}");
        }
    }
    // The built-in BLAS declares provider-owned threading, so a forced
    // sequential batch is a safety constraint the strategy cannot override.
    let mut sequential = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Sequential));
    let error = run(&mut sequential).unwrap_err();
    assert_eq!(
        error.kind(),
        tenferro_tensor::ErrorKind::Unsupported,
        "{error}"
    );
}
