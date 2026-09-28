//! Same-revision route comparison for strided batched contractions on the BLAS
//! provider: the whole-batch vendor call (`cblas_?gemm_batch`) against one
//! provider GEMM per batch item.
//!
//! Run with one backend thread and one provider thread, for example
//! `OPENBLAS_NUM_THREADS=1 cargo bench -p tenferro-cpu --features blas-openblas
//! --bench strided_batch_route`. Every strategy's output is checked against the
//! per-item route before timing.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use tenferro_cpu::{CpuBackend, CpuBackendKind, CpuBatchPolicy, CpuBatchStrategy};
use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead, TensorWrite};

/// `(m, k, n, batch)` for `[m, k, batch] x [k, n, batch] -> [m, n, batch]`.
const CASES: [(usize, usize, usize, usize); 3] =
    [(4, 4, 4, 2048), (8, 8, 8, 512), (16, 16, 16, 128)];

const STRATEGIES: [(&str, CpuBatchStrategy); 3] = [
    ("auto", CpuBatchStrategy::Auto),
    ("whole_batch_vendor", CpuBatchStrategy::WholeBatchVendor),
    ("provider_items", CpuBatchStrategy::ProviderItems),
];

fn deterministic_data(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|idx| ((idx * 17 + seed * 31 + 7) % 101) as f64 / 101.0 - 0.5)
        .collect()
}

fn batched_matmul_config() -> DotGeneralConfig {
    DotGeneralConfig {
        lhs_contracting_dims: [1].as_slice().into(),
        rhs_contracting_dims: [0].as_slice().into(),
        lhs_batch_dims: [2].as_slice().into(),
        rhs_batch_dims: [2].as_slice().into(),
    }
}

fn backend(strategy: CpuBatchStrategy) -> CpuBackend {
    let backend = CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas)
        .expect("one-thread BLAS backend")
        .with_batch_policy(CpuBatchPolicy::new(strategy));
    assert_eq!(backend.num_threads(), 1);
    backend
}

fn run_into(
    backend: &mut CpuBackend,
    lhs: &Tensor,
    rhs: &Tensor,
    config: &DotGeneralConfig,
    out: &mut Tensor,
) {
    backend
        .with_backend_session(|session| {
            session.dot_general_read_into(
                TensorRead::from_tensor(lhs),
                TensorRead::from_tensor(rhs),
                config,
                TensorWrite::from_tensor(out),
            )
        })
        .expect("session entry")
        .expect("strided batched contraction");
}

fn bench_strided_batch_route(c: &mut Criterion) {
    let config = batched_matmul_config();
    let mut group = c.benchmark_group("strided_batch_route/f64_1t");
    for (m, k, n, batch) in CASES {
        let lhs =
            Tensor::from_vec_col_major(vec![m, k, batch], deterministic_data(m * k * batch, 1))
                .expect("lhs");
        let rhs =
            Tensor::from_vec_col_major(vec![k, n, batch], deterministic_data(k * n * batch, 2))
                .expect("rhs");
        let zeros = || Tensor::from_vec_col_major(vec![m, n, batch], vec![0.0; m * n * batch]);

        let mut reference = zeros().expect("reference output");
        run_into(
            &mut backend(CpuBatchStrategy::ProviderItems),
            &lhs,
            &rhs,
            &config,
            &mut reference,
        );
        let reference = reference.as_slice::<f64>().expect("host output").to_vec();

        let case = format!("{m}x{k}x{n}_b{batch}");
        for (name, strategy) in STRATEGIES {
            let mut backend = backend(strategy);
            let mut out = zeros().expect("timed output");
            run_into(&mut backend, &lhs, &rhs, &config, &mut out);
            let got = out.as_slice::<f64>().expect("host output");
            for (lhs_value, rhs_value) in got.iter().zip(&reference) {
                assert!(
                    (lhs_value - rhs_value).abs() <= 1e-12 * rhs_value.abs().max(1.0),
                    "{name} disagrees with the per-item route for {case}"
                );
            }
            group.bench_function(BenchmarkId::new(name, &case), |bench| {
                bench.iter(|| {
                    run_into(
                        &mut backend,
                        black_box(&lhs),
                        black_box(&rhs),
                        &config,
                        &mut out,
                    );
                    black_box(&out);
                });
            });
        }
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .warm_up_time(std::time::Duration::from_secs(1))
        .measurement_time(std::time::Duration::from_secs(3))
        .sample_size(50)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_strided_batch_route
}
criterion_main!(benches);
