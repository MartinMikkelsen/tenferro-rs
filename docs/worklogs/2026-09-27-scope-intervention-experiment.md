# Execution-scope intervention experiment (#1926 / #1929 Phase C)

**Date**: 2026-09-27
**Branch**: `refactor/1929-session-route-unification` (head `eec3d9591`)
**Question**: does wrapping an eager backward pass in one execution scope remove
the remaining per-operation session cost, or is the residue something else?

## Predeclared bar and protocol

The bar was fixed before the first measurement, per the council/Astra review:
recover **at least 80%** of the measured `linalg_vjp_gate` delta. The council's
chair asked for an intervention experiment that keeps the scope open/close *inside*
the timed region and does not amortize one scope over all Criterion iterations.

* Bench-only: `crates/tenferro-linalg/benches/linalg_vjp_gate.rs` builds the fixture
  runtime from a `CpuBackend` it also keeps a clone of, and adds
  `<op>_vjp_scope` cases next to the existing ones. No library file changed.
* Both variants are cases of the **same** benchmark binary and the same run, so the
  comparison is matched by construction (no cross-run drift between the two sides).
* `bench.iter(|| backend.with_execution_scope(|| run_vjp(black_box(&fixture))).unwrap())`
  — one scope per iteration, including open/close in the timed region.
* Environment: `taskset -c 1`, `RAYON/OMP/OPENBLAS/MKL/VECLIB/NUMEXPR=1`,
  `TENFERRO_BENCH_THREADS=1`, Criterion defaults (`--warm-up-time 2
  --measurement-time 5 --sample-size 100`), `TENFERRO_LINALG_VJP_BENCH_SIZES=8` (and
  16 for one confirmation run).
* Validity: the pinned core was sampled for five seconds immediately before and
  after every run; all reads were 0% busy. Three runs at size 8, one at size 16.

## Result

Medians, size 8 (three runs) and size 16 (one run); baselines from
`docs/testing/session-route-baseline-recaptured.json`:

| case | baseline | no scope | with scope | speedup vs no scope | vs baseline |
| --- | --- | --- | --- | --- | --- |
| `triangular_solve_vjp/8` | 320.95 µs | 379.84 µs (+18.3%) | **178.79 µs** | 2.12× | 1.80× |
| `svd_values_vjp/8` | 336.29 µs | 370.81 µs (+10.3%) | **182.78 µs** | 2.03× | 1.84× |
| `triangular_solve_vjp/16` | 319.77 µs | 383.43 µs (+19.9%) | **185.01 µs** | 2.07× | 1.73× |
| `svd_values_vjp/16` | 333.32 µs | 372.11 µs (+11.6%) | **185.15 µs** | 2.01× | 1.80× |

Run-to-run spread at size 8 was small on both sides (no scope 362–379 µs, scope
181–187 µs), and the effect (~200 µs) is two orders of magnitude above that spread.
Recovery is far beyond the 80% bar: the scope does not merely recover the
regression, it makes the VJP about **1.8× faster than the pre-unification
baseline**.

Logs: `target/scope-intervention/run{1,2,3,16}.log`.

## Count diagnostic (Astra's ask)

`TENFERRO_PROFILE_CPU_SESSION=1` with `TENFERRO_PROFILE_CPU_SESSION_PRINT_EVERY=2000`,
same case, both variants:

| section (per 2000-call window) | no scope | with scope |
| --- | --- | --- |
| `with_backend_session_cached.total` | 2000 calls, **18.92 µs/call** | 2000 calls, **7.90 µs/call** |
| `with_backend_session_cached.exec_session` | 2000 calls, 18.78 µs/call | 2000 calls, 7.78 µs/call |
| `with_backend_session_cached.exec_body` | 12000 calls, 4.00 µs/call | 12000 calls, 3.59 µs/call |
| `with_backend_session_cached.session_construct` | 12000 calls, 0.032 µs/call | 12000 calls, 0.032 µs/call |

Two conclusions:

* the **entry counts are identical** (2000 outer, 12000 inner per window) — the
  scope changes the *cost per entry*, not the number of entries: the outer entry
  drops from 18.9 µs to 7.9 µs once the permit and pool loan are already held;
* session construction inside the scope is already ~32 ns, which is the same order
  as the in-scope entry cost measured in Phase A (0.17 µs), and confirms that the
  fresh-session path was paying for permit acquisition and pool/resource setup
  rather than for the struct itself.

The sections do not account for the whole ~200 µs per VJP: `session_construct` is
recorded *after* permit acquisition and executor entry, and the resource lookup in
`with_execution_resources` happens before it, so a complete attribution needs more
instrumentation. The profiler is a diagnostic aid here, not a ledger.

## Decision

* The scope hypothesis is **confirmed**, well beyond the predeclared bar; the
  direction is worth adopting.
* Astra's ordering stands: **design the execution-ownership protocol before freezing
  any public API**. The protocol must name the holder of the permit and the entered
  execution, the release boundary on normal/error/unwind paths, joining behaviour
  when a user has already opened a scope, behaviour for other backends and external
  executors, and the waiting relationships under contention. The CPU scope currently
  rejects a second scope or active execution with `Error::RuntimeState`, so a hook
  whose contract "always returns `R`" has to say how that case is surfaced.
* This is a **cost reduction, not only a regression fix**: the measurement suggests
  the pre-unification eager path was paying much of the same per-operation session
  cost, so the win is available on top of the unification rather than as a
  restoration of it.

## Limits

* Bench-level scope: it exercises the scope as a *caller* would, which is a
  legitimate and documented pattern (`session_chain`'s scope arm). It does not
  demonstrate the library opening the scope itself.
* Only `linalg_vjp_gate` sizes 8 and 16, f64, one host, one pinned core, one
  three-run repetition at size 8. The eager small-op cases
  (`eager_dispatch_baseline`) were not re-measured under a scope; they are the other
  half of the residue and should be included in the follow-up.
* The single-case criterion runs here are the practical substitute for the
  repository's three alternating baseline/candidate pairs on this contended host;
  the certification protocol still applies to the final performance claim.
