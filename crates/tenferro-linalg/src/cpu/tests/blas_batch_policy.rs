//! The LAPACK packed-LU loop honors only the batch strategies that describe
//! it and rejects the rest with a typed error (#1938 D9, #1884).

use super::*;
use tenferro_cpu::{CpuBackendKind, CpuBatchPolicy, CpuBatchStrategy};

fn diagonally_dominant(n: usize, batch: usize) -> Tensor {
    let data = (0..n * n * batch)
        .map(|index| {
            let (row, col) = (index % n, (index / n) % n);
            if row == col {
                n as f64 + 1.0
            } else {
                ((index % 5) as f64) * 0.1
            }
        })
        .collect::<Vec<_>>();
    Tensor::from_vec_col_major(vec![n, n, batch], data).unwrap()
}

fn blas_backend(strategy: CpuBatchStrategy) -> CpuBackend {
    CpuBackend::with_threads_and_kind(4, CpuBackendKind::Blas)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(strategy))
}

#[test]
fn blas_packed_lu_accepts_auto_and_provider_items_and_rejects_the_rest() {
    let (n, batch) = (4, 3);
    let b = || Tensor::from_vec_col_major(vec![n, batch], vec![1.0_f64; n * batch]).unwrap();
    for strategy in [CpuBatchStrategy::Auto, CpuBatchStrategy::ProviderItems] {
        let mut backend = blas_backend(strategy);
        let factors = with_cpu_linalg(&mut backend, |s| {
            s.lu_factor(&diagonally_dominant(n, batch))
        })
        .unwrap();
        with_cpu_linalg(&mut backend, |s| {
            s.lu_solve_prepared(
                &diagonally_dominant(n, batch),
                &factors[0],
                &factors[1],
                &b(),
                false,
                false,
            )
        })
        .unwrap();
        with_cpu_linalg(&mut backend, |s| {
            s.lu_factor_solve(&diagonally_dominant(n, batch), &b())
        })
        .unwrap();
    }

    let mut auto = blas_backend(CpuBatchStrategy::Auto);
    let factors =
        with_cpu_linalg(&mut auto, |s| s.lu_factor(&diagonally_dominant(n, batch))).unwrap();
    for strategy in [
        CpuBatchStrategy::Sequential,
        CpuBatchStrategy::OuterParallel,
        CpuBatchStrategy::WholeBatchVendor,
    ] {
        let mut backend = blas_backend(strategy);
        let errors = [
            with_cpu_linalg(&mut backend, |s| {
                s.lu_factor(&diagonally_dominant(n, batch))
            })
            .unwrap_err(),
            with_cpu_linalg(&mut backend, |s| {
                s.lu_solve_prepared(
                    &diagonally_dominant(n, batch),
                    &factors[0],
                    &factors[1],
                    &b(),
                    false,
                    false,
                )
            })
            .unwrap_err(),
            with_cpu_linalg(&mut backend, |s| {
                s.lu_factor_solve(&diagonally_dominant(n, batch), &b())
            })
            .unwrap_err(),
        ];
        for error in errors {
            assert_eq!(
                error.kind(),
                tenferro_tensor::ErrorKind::Unsupported,
                "{strategy:?}: {error}"
            );
            assert!(
                error.to_string().contains(&format!("{strategy:?}")),
                "{error}"
            );
        }
        // One matrix is not a batch: the policy does not apply.
        with_cpu_linalg(&mut backend, |s| s.lu_factor(&diagonally_dominant(n, 1))).unwrap();
    }
}
