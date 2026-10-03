//! The tprims provider no longer implements linear algebra (tprims dropped
//! `trsm` and its linalg crate), so with it installed the built-in
//! `tenferro-linalg` kernels must still run every factorization and solve and
//! give the default backend's result.
use std::sync::Arc;

use tenferro_cpu::{
    with_cpu_exec_session, CpuBackend, CpuBackendKind, CpuExecSession, CpuProviderBundle,
};
use tenferro_cpu_tprims::TprimsProvider;
use tenferro_linalg::backend::LinalgBackend;
use tenferro_tensor::{BackendSessionHost, Tensor};

fn backend(tprims: bool) -> CpuBackend {
    let base = CpuBackend::with_threads(2).unwrap();
    if !tprims {
        return base;
    }
    let builder = CpuProviderBundle::builder(CpuBackendKind::default_compiled())
        .gemm_provider(Arc::new(TprimsProvider::new()))
        .prefer_general_contraction_provider(Arc::new(TprimsProvider::new()));
    base.with_provider_bundle(builder.build().unwrap()).unwrap()
}

fn run<R: Send>(b: &mut CpuBackend, f: impl FnOnce(&mut CpuExecSession<'_>) -> R + Send) -> R {
    b.with_backend_session(|s| with_cpu_exec_session(s, f).expect("a CPU backend session"))
        .unwrap()
}

fn close(what: &str, got: &Tensor, want: &Tensor) {
    assert_eq!(got.shape(), want.shape(), "{what} shape");
    let (g, w) = (
        got.as_slice::<f64>().unwrap(),
        want.as_slice::<f64>().unwrap(),
    );
    for (i, (x, y)) in g.iter().zip(w).enumerate() {
        assert!(
            (x - y).abs() <= 1e-10 * (1.0 + y.abs()),
            "{what}[{i}]: {x} vs {y}"
        );
    }
}

/// Symmetric positive definite `n x n`: `M Mᵀ + n I`.
fn spd(n: usize) -> Tensor {
    let m: Vec<f64> = (0..n * n)
        .map(|i| ((i * 7 + 3) % 11) as f64 / 5.0 - 1.0)
        .collect();
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] = (0..n).map(|l| m[i + l * n] * m[j + l * n]).sum::<f64>()
                + if i == j { n as f64 } else { 0.0 };
        }
    }
    Tensor::from_vec_col_major(vec![n, n], a).unwrap()
}

#[test]
fn linalg_runs_on_the_builtin_kernels_with_the_provider_installed() {
    let (mut d, mut t) = (backend(false), backend(true));
    let a = spd(6);
    let b =
        Tensor::from_vec_col_major(vec![6, 3], (0..18).map(|i| i as f64 * 0.25).collect()).unwrap();
    let (w, g) = (
        run(&mut d, |c| c.cholesky(&a)).unwrap(),
        run(&mut t, |c| c.cholesky(&a)).unwrap(),
    );
    close("cholesky", &g, &w);
    close(
        "solve",
        &run(&mut t, |c| c.solve(&a, &b)).unwrap(),
        &run(&mut d, |c| c.solve(&a, &b)).unwrap(),
    );
    for bits in 0..8u32 {
        let (lower, trans, unit) = (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0);
        close(
            &format!("triangular_solve {bits:03b}"),
            &run(&mut t, |c| {
                c.triangular_solve(&w, &b, true, lower, trans, unit)
            })
            .unwrap(),
            &run(&mut d, |c| {
                c.triangular_solve(&w, &b, true, lower, trans, unit)
            })
            .unwrap(),
        );
    }
    close(
        "eigh_values",
        &run(&mut t, |c| c.eigh_values(&a)).unwrap(),
        &run(&mut d, |c| c.eigh_values(&a)).unwrap(),
    );
    close(
        "svd_values",
        &run(&mut t, |c| c.svd_values(&a)).unwrap(),
        &run(&mut d, |c| c.svd_values(&a)).unwrap(),
    );
    let (w, g) = (
        run(&mut d, |c| c.qr(&b)).unwrap(),
        run(&mut t, |c| c.qr(&b)).unwrap(),
    );
    for (i, (x, y)) in g.iter().zip(&w).enumerate() {
        close(&format!("qr[{i}]"), x, y);
    }
}
