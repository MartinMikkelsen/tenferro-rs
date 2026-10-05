//! Issue #1884: the faer batch lanes must reproduce the serial batch.
//!
//! These cases drive the batched faer kernels on a four-thread backend, so the
//! lane branches in `faer_linalg/packed_lu.rs` run here even though the resident
//! oracle sweeps use one thread.

use super::*;
use tenferro_cpu::CpuBackendKind;

fn faer_tensor(shape: &[usize], data: &[f64]) -> Tensor {
    Tensor::from_vec_col_major(shape.to_vec(), data.to_vec()).unwrap()
}

/// Cyclic dominant matrices, so every batch member needs pivoting.
fn pivoting(n: usize, batch: usize, seed: u64) -> Vec<f64> {
    let mut values: Vec<f64> = (0..n * n * batch)
        .map(|index| {
            let value = (index as u64)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(seed);
            ((value % 4096) as f64 / 2048.0) - 1.0
        })
        .collect();
    for matrix in 0..batch {
        for col in 0..n {
            values[matrix * n * n + col + col * n] += n as f64 + 2.0;
        }
    }
    values
}

fn rhs(n: usize, nrhs: usize, batch: usize) -> Vec<f64> {
    (0..n * nrhs * batch)
        .map(|index| ((index % 7) as f64) - 3.0)
        .collect()
}

/// Numeric contents as `f64`, so factor, pivot, and solution tensors compare
/// with the same helper.
fn data(tensor: &Tensor) -> Vec<f64> {
    match tensor.dtype() {
        DType::F64 => tensor
            .as_typed::<f64>()
            .unwrap()
            .host_data()
            .unwrap()
            .to_vec(),
        DType::I32 => tensor
            .as_typed::<i32>()
            .unwrap()
            .host_data()
            .unwrap()
            .iter()
            .map(|&value| f64::from(value))
            .collect(),
        other => panic!("unexpected dtype {other:?}"),
    }
}

/// A four-thread batch must agree with a one-thread batch bit for bit.
///
/// Each lane factorizes one matrix at a time with `Par::Seq`, so the per-matrix
/// arithmetic is identical to the serial loop and the results must match
/// exactly, not approximately.
#[test]
fn batched_faer_lanes_reproduce_the_serial_batch() {
    let n = 8usize;
    let batch = 8usize;
    let a_data = pivoting(n, batch, 5);
    let b_data = rhs(n, 1, batch);
    let a = || faer_tensor(&[n, n, batch], &a_data);
    let b = || faer_tensor(&[n, batch], &b_data);

    let mut serial = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    let mut parallel = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer).unwrap();

    let serial_factors = with_cpu_linalg(&mut serial, |s| s.lu_factor(&a())).unwrap();
    let parallel_factors = with_cpu_linalg(&mut parallel, |s| s.lu_factor(&a())).unwrap();
    assert_eq!(data(&parallel_factors[0]), data(&serial_factors[0]));
    assert_eq!(data(&parallel_factors[1]), data(&serial_factors[1]));

    let serial_x = with_cpu_linalg(&mut serial, |s| {
        s.lu_solve_prepared(
            &a(),
            &serial_factors[0],
            &serial_factors[1],
            &b(),
            false,
            false,
        )
    })
    .unwrap();
    let parallel_x = with_cpu_linalg(&mut parallel, |s| {
        s.lu_solve_prepared(
            &a(),
            &parallel_factors[0],
            &parallel_factors[1],
            &b(),
            false,
            false,
        )
    })
    .unwrap();
    assert_eq!(data(&parallel_x), data(&serial_x));

    let serial_fused = with_cpu_linalg(&mut serial, |s| s.lu_factor_solve(&a(), &b())).unwrap();
    let parallel_fused = with_cpu_linalg(&mut parallel, |s| s.lu_factor_solve(&a(), &b())).unwrap();
    for (parallel_part, serial_part) in parallel_fused.iter().zip(&serial_fused) {
        assert_eq!(data(parallel_part), data(serial_part));
    }

    // A zero-column RHS exercises the factor-only lane path.
    let empty = faer_tensor(&[n, 0, batch], &[]);
    let serial_only = with_cpu_linalg(&mut serial, |s| s.lu_factor_solve(&a(), &empty)).unwrap();
    let parallel_only =
        with_cpu_linalg(&mut parallel, |s| s.lu_factor_solve(&a(), &empty)).unwrap();
    for (parallel_part, serial_part) in parallel_only.iter().zip(&serial_only) {
        assert_eq!(data(parallel_part), data(serial_part));
    }
}

/// A singular batch member keeps reporting the same typed error with lanes.
#[test]
fn batched_faer_lanes_report_a_singular_member() {
    let n = 8usize;
    let batch = 8usize;
    for threads in [1usize, 4] {
        let mut a_data = pivoting(n, batch, 3);
        // Make the last matrix exactly singular.
        let offset = (batch - 1) * n * n;
        for index in 0..n * n {
            a_data[offset + index] = 0.0;
        }
        let a = faer_tensor(&[n, n, batch], &a_data);
        let b = faer_tensor(&[n, batch], &rhs(n, 1, batch));
        let mut backend = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Faer).unwrap();
        let err = with_cpu_linalg(&mut backend, |s| s.lu_factor_solve(&a, &b)).unwrap_err();
        assert!(
            err.to_string().contains("singular"),
            "{threads}T: expected a singular error, got {err}"
        );
    }
}

/// Every forced batch strategy with a route reproduces the one-thread batch,
/// and a strategy without a route fails with a typed error (#1938 D9).
#[test]
fn forced_batch_strategies_reproduce_the_serial_batch_or_fail_typed() {
    use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};

    let n = 8usize;
    let batch = 8usize;
    let a_data = pivoting(n, batch, 11);
    let a = || faer_tensor(&[n, n, batch], &a_data);
    let mut serial = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
    let reference = with_cpu_linalg(&mut serial, |s| s.lu_factor(&a())).unwrap();

    for strategy in [
        CpuBatchStrategy::Sequential,
        CpuBatchStrategy::ProviderItems,
        CpuBatchStrategy::OuterParallel,
    ] {
        let mut backend = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)
            .unwrap()
            .with_batch_policy(CpuBatchPolicy::new(strategy));
        let factors = with_cpu_linalg(&mut backend, |s| s.lu_factor(&a())).unwrap();
        // Sequential and outer lanes run each item with `Par::Seq`, which is
        // bit-identical to the serial loop; provider items may use faer's own
        // parallelism, which changes nothing for pivoted LU of this size.
        assert_eq!(data(&factors[0]), data(&reference[0]), "{strategy:?}");
        assert_eq!(data(&factors[1]), data(&reference[1]), "{strategy:?}");
    }

    for (threads, strategy) in [
        (1, CpuBatchStrategy::OuterParallel),
        (4, CpuBatchStrategy::WholeBatchVendor),
    ] {
        let mut backend = CpuBackend::with_threads_and_kind(threads, CpuBackendKind::Faer)
            .unwrap()
            .with_batch_policy(CpuBatchPolicy::new(strategy));
        let error = with_cpu_linalg(&mut backend, |s| s.lu_factor(&a())).unwrap_err();
        assert_eq!(
            error.kind(),
            tenferro_tensor::ErrorKind::Unsupported,
            "{strategy:?}: {error}"
        );
    }
}

/// `(lanes, item runs with the context's pool)` the faer route resolves for one batched call.
fn resolved_plan(
    backend: &mut CpuBackend,
    op: tlinalg::Op,
    batch: usize,
    max_item_dim: Option<usize>,
) -> (usize, bool) {
    with_cpu_linalg(backend, |session| {
        session.with_linalg_pool(|context, _| {
            let plan = match max_item_dim {
                Some(dim) => crate::cpu::tlinalg::lane_plan_for_item(context, op, batch, dim)?,
                None => crate::cpu::tlinalg::lane_plan(context, op, batch)?,
            };
            Ok((
                plan.lanes,
                matches!(plan.item_parallel, tlinalg::Parallel::Pool { .. }),
            ))
        })
    })
    .unwrap()
}

/// The provisional `Auto` guard (#2000): families other than packed LU fan a batch out only for
/// small items; large items keep one lane with the context's parallelism, as before the batched
/// extraction. Forced strategies and the packed-LU family are unaffected.
#[test]
fn auto_fans_out_only_small_items_outside_packed_lu() {
    use crate::cpu::tlinalg::AUTO_FAN_OUT_MAX_ITEM_DIM;
    use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};
    use tlinalg::Op;

    let mut auto = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer).unwrap();
    // Many tiny matrices: Auto fans out over the budget.
    assert_eq!(resolved_plan(&mut auto, Op::Eigh, 1024, Some(4)), (4, true));
    // Exactly at the guard is still small.
    let at_guard = resolved_plan(&mut auto, Op::Svd, 1024, Some(AUTO_FAN_OUT_MAX_ITEM_DIM));
    assert_eq!(at_guard.0, 4);
    // Large matrices: one lane, items keep the context's parallelism.
    assert_eq!(resolved_plan(&mut auto, Op::Eigh, 8, Some(128)), (1, true));
    assert_eq!(
        resolved_plan(&mut auto, Op::Qr, 8, Some(AUTO_FAN_OUT_MAX_ITEM_DIM + 1)),
        (1, true)
    );
    // The packed-LU family keeps its plan at any item size.
    assert_eq!(resolved_plan(&mut auto, Op::LuFactor, 8, None), (4, true));

    // A forced outer fan-out still fans out a batch of large items.
    let mut outer = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::OuterParallel));
    assert_eq!(
        resolved_plan(&mut outer, Op::Eigh, 8, Some(128)),
        (4, false)
    );
    // Forced sequential and provider-items behave as before for every size.
    let mut sequential = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Sequential));
    assert_eq!(
        resolved_plan(&mut sequential, Op::Eigh, 1024, Some(4)),
        (1, false)
    );
    let mut provider = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer)
        .unwrap()
        .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::ProviderItems));
    assert_eq!(
        resolved_plan(&mut provider, Op::Eigh, 1024, Some(4)),
        (1, true)
    );
}

/// A large-item batch under `Auto` and a small-item fan-out both reproduce the serial results.
#[test]
fn guarded_auto_batches_match_the_serial_results() {
    for (n, batch) in [(4usize, 64usize), (72, 2)] {
        let a = faer_tensor(&[n, n, batch], &pivoting(n, batch, 7));
        let mut serial = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).unwrap();
        let mut parallel = CpuBackend::with_threads_and_kind(4, CpuBackendKind::Faer).unwrap();
        let reference = with_cpu_linalg(&mut serial, |s| s.lu(&a)).unwrap();
        let actual = with_cpu_linalg(&mut parallel, |s| s.lu(&a)).unwrap();
        for (actual, reference) in actual.iter().zip(&reference) {
            for (x, y) in data(actual).iter().zip(&data(reference)) {
                assert!(
                    (x - y).abs() <= 1e-12 * (1.0 + y.abs()),
                    "n={n}: {x} vs {y}"
                );
            }
        }
    }
}
