//! Allocation accounting for the packed-LU family on the CPU routes.
//!
//! The packed-LU family (`lu_factor`, `lu_solve_prepared`, `lu_factor_solve`) is the first slice
//! being moved out of `tenferro-linalg` into an extracted crate, so it needs a recorded
//! steady-state allocation baseline on both routes *before* anything moves. The ceilings below are
//! what the current implementation achieves; they exist to catch a move that quietly starts
//! reacquiring scratch per call.
//!
//! This target owns the process allocator, so it must stay a separate test binary. Counts are
//! steady-state: the pool is primed by warm-up calls first, because a cold first call legitimately
//! allocates.
//!
//! Byte totals are printed rather than asserted. The *number* of allocations is structural; the
//! sizes are not, because they depend on the shapes a caller chooses.

#![cfg(feature = "cpu-faer")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use num_complex::Complex64;
use tenferro_cpu::{with_cpu_exec_session, CpuBackend, CpuBackendKind};
use tenferro_linalg::LinalgBackend;
use tenferro_tensor::{BackendSession, BackendSessionHost, Tensor, TypedTensor};

/// Counts allocations and live bytes while armed.
struct CountingAllocator;

static ARMED: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);
static LIVE_BYTES: AtomicUsize = AtomicUsize::new(0);
static PEAK_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every method forwards to the system allocator with the caller's original layout and
// pointer; the counters only observe.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: `layout` is the caller's, forwarded unchanged.
        let pointer = unsafe { System.alloc(layout) };
        if !pointer.is_null() && ARMED.load(Ordering::Relaxed) {
            record_allocation(layout.size());
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        if ARMED.load(Ordering::Relaxed) {
            LIVE_BYTES.fetch_sub(
                layout.size().min(LIVE_BYTES.load(Ordering::Relaxed)),
                Ordering::Relaxed,
            );
        }
        // SAFETY: `pointer`/`layout` are the caller's, forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: all three arguments are the caller's, forwarded unchanged.
        let new_pointer = unsafe { System.realloc(pointer, layout, new_size) };
        if !new_pointer.is_null() && ARMED.load(Ordering::Relaxed) && new_size > layout.size() {
            record_allocation(new_size - layout.size());
        }
        new_pointer
    }
}

fn record_allocation(size: usize) {
    ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
    ALLOCATED_BYTES.fetch_add(size, Ordering::Relaxed);
    let live = LIVE_BYTES.fetch_add(size, Ordering::Relaxed) + size;
    PEAK_BYTES.fetch_max(live, Ordering::Relaxed);
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct AllocationReport {
    allocations: usize,
    allocated_bytes: usize,
    peak_live_bytes: usize,
}

/// Measure one call with the counters armed.
///
/// Single-threaded by construction: the backends below are built with one thread, and the counters
/// are process-wide.
fn measure(operation: impl FnOnce()) -> AllocationReport {
    ALLOCATIONS.store(0, Ordering::Relaxed);
    ALLOCATED_BYTES.store(0, Ordering::Relaxed);
    LIVE_BYTES.store(0, Ordering::Relaxed);
    PEAK_BYTES.store(0, Ordering::Relaxed);
    ARMED.store(true, Ordering::Relaxed);
    operation();
    ARMED.store(false, Ordering::Relaxed);
    AllocationReport {
        allocations: ALLOCATIONS.load(Ordering::Relaxed),
        allocated_bytes: ALLOCATED_BYTES.load(Ordering::Relaxed),
        peak_live_bytes: PEAK_BYTES.load(Ordering::Relaxed),
    }
}

fn faer_backend() -> CpuBackend {
    CpuBackend::with_threads_and_kind(1, CpuBackendKind::Faer).expect("faer CPU backend")
}

#[cfg(feature = "cpu-blas")]
fn blas_backend() -> CpuBackend {
    CpuBackend::with_threads_and_kind(1, CpuBackendKind::Blas).expect("BLAS CPU backend")
}

fn sample_real(n: usize) -> Vec<f64> {
    (0..n * n)
        .map(|index| {
            let row = (index % n) as f64;
            let col = (index / n) as f64;
            if row == col {
                n as f64 + 1.0
            } else {
                1.0 + row * 0.25 - col * 0.5
            }
        })
        .collect()
}

fn f64_matrix(n: usize) -> Tensor {
    Tensor::from_typed::<f64>(TypedTensor::from_vec_col_major(vec![n, n], sample_real(n)).unwrap())
}

fn c64_matrix(n: usize) -> Tensor {
    let data = sample_real(n)
        .into_iter()
        .enumerate()
        .map(|(index, real)| {
            let imag = if index % (n + 1) == 0 { 0.5 } else { -0.25 };
            Complex64::new(real, imag)
        })
        .collect();
    Tensor::from_typed::<Complex64>(TypedTensor::from_vec_col_major(vec![n, n], data).unwrap())
}

/// Run `operation` until the report stops shrinking, so pooled scratch is already warm.
fn steady_state(
    host: &mut CpuBackend,
    operation: impl Fn(&mut dyn BackendSession) + Send + Sync,
) -> AllocationReport {
    let mut best: Option<AllocationReport> = None;
    for _ in 0..4 {
        let report = host
            .with_backend_session(|session| measure(|| operation(session)))
            .unwrap();
        if best.is_none_or(|best| report.allocations < best.allocations) {
            best = Some(report);
        }
    }
    best.unwrap_or_default()
}

fn with_session<T>(
    session: &mut dyn BackendSession,
    operation: impl FnOnce(&mut tenferro_cpu::CpuExecSession<'_>) -> T,
) -> T {
    with_cpu_exec_session(session, operation).expect("host exposes a CPU exec session")
}

/// Steady-state allocation ceiling per route and case.
///
/// These are the counts the current implementation achieves, not aspirations. They exist to stop a
/// move from quietly reacquiring per-call scratch. `lu_factor` returns `(packed_lu, pivots,
/// parity)`, so its wall is higher than a single-output primitive by construction.
const F64_LU_CEILINGS: &[(&str, usize)] = &[
    ("faer/lu_factor/48", 11),
    ("faer/lu_solve_prepared/48", 5),
    ("faer/lu_factor_solve/48", 13),
    ("blas/lu_factor/48", 6),
    ("blas/lu_solve_prepared/48", 5),
    ("blas/lu_factor_solve/48", 8),
    ("faer/lu_factor/complex32", 11),
    ("blas/lu_factor/complex32", 6),
];

fn ceiling(name: &str) -> usize {
    F64_LU_CEILINGS
        .iter()
        .find(|(case, _)| *case == name)
        .map(|(_, ceiling)| *ceiling)
        .unwrap_or_else(|| panic!("no recorded allocation ceiling for {name}"))
}

fn check(
    host: &mut CpuBackend,
    failures: &mut Vec<String>,
    name: &str,
    operation: impl Fn(&mut dyn BackendSession) + Send + Sync,
) {
    let report = steady_state(host, operation);
    eprintln!(
        "{name}: {} allocations, {} bytes, peak {} live bytes",
        report.allocations, report.allocated_bytes, report.peak_live_bytes
    );
    let ceiling = ceiling(name);
    if report.allocations > ceiling {
        failures.push(format!(
            "{name}: {} allocations exceeds the recorded ceiling of {ceiling} \
             ({} bytes, peak {} live bytes)",
            report.allocations, report.allocated_bytes, report.peak_live_bytes
        ));
    }
}

#[test]
fn packed_lu_routes_take_their_scratch_from_the_session_buffer_pool() {
    let mut failures = Vec::new();

    let mut faer = faer_backend();
    let a = f64_matrix(48);
    let packed = faer
        .with_backend_session(|session| with_session(session, |cpu| cpu.lu_factor(&a).unwrap()))
        .unwrap();
    check(&mut faer, &mut failures, "faer/lu_factor/48", |session| {
        with_session(session, |cpu| {
            cpu.lu_factor(&a).unwrap();
        });
    });
    check(
        &mut faer,
        &mut failures,
        "faer/lu_solve_prepared/48",
        |session| {
            with_session(session, |cpu| {
                cpu.lu_solve_prepared(&a, &packed[0], &packed[1], &a, false, false)
                    .unwrap();
            });
        },
    );
    check(
        &mut faer,
        &mut failures,
        "faer/lu_factor_solve/48",
        |session| {
            with_session(session, |cpu| {
                cpu.lu_factor_solve(&a, &a).unwrap();
            });
        },
    );

    let c = c64_matrix(32);
    check(
        &mut faer,
        &mut failures,
        "faer/lu_factor/complex32",
        |session| {
            with_session(session, |cpu| {
                cpu.lu_factor(&c).unwrap();
            });
        },
    );

    #[cfg(feature = "cpu-blas")]
    {
        let mut blas = blas_backend();
        let packed = blas
            .with_backend_session(|session| with_session(session, |cpu| cpu.lu_factor(&a).unwrap()))
            .unwrap();
        check(&mut blas, &mut failures, "blas/lu_factor/48", |session| {
            with_session(session, |cpu| {
                cpu.lu_factor(&a).unwrap();
            });
        });
        check(
            &mut blas,
            &mut failures,
            "blas/lu_solve_prepared/48",
            |session| {
                with_session(session, |cpu| {
                    cpu.lu_solve_prepared(&a, &packed[0], &packed[1], &a, false, false)
                        .unwrap();
                });
            },
        );
        check(
            &mut blas,
            &mut failures,
            "blas/lu_factor_solve/48",
            |session| {
                with_session(session, |cpu| {
                    cpu.lu_factor_solve(&a, &a).unwrap();
                });
            },
        );
        check(
            &mut blas,
            &mut failures,
            "blas/lu_factor/complex32",
            |session| {
                with_session(session, |cpu| {
                    cpu.lu_factor(&c).unwrap();
                });
            },
        );
    }

    assert!(
        failures.is_empty(),
        "packed-LU routes allocate more per call than the recorded ceilings:\n{}",
        failures.join("\n")
    );
}
