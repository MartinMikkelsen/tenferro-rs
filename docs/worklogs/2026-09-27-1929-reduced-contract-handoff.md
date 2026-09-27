# #1929: explicit-session / route-API handoff (reduced contract)

**Branch:** `refactor/1929-session-route-unification`

**Contract:** [#1929's 2026-09-27 revision](https://github.com/tensor4all/tenferro-rs/issues/1929). This is a delivery and verification record, **not** a performance certification. No new measurement was made for this handoff.

## Delivered

- Phase B deleted the 31 paired backend owner one-shots, moved CPU/CUDA/WebGPU operation bodies to sessions, threaded runtime and extension routes through borrowed sessions, and preserved the 13 required one-shots without `_read` siblings. The compiled instruction path shares one session across its probe, execution and last-use reclaim; an impossible probe makes no entry. `scripts/audit-session-entry.py --check` enforces the reviewed library entry allowlist; `with_evaluation_scope` has **zero** entries.
- `EagerRuntime::with_eager_session` lends a runtime-bound `EagerSession` for standard eager operations, leaf import, host ingress and value duplication. The implicit `EagerTensor` operation methods/operators were removed rather than forwarding to a per-op session. Borrowed operations check runtime identity and keep untracked-input semantic recording and lazy-view materialization inside the active session. Gradient-slot accumulation borrows one session at the backward boundary. `EagerBackend` no longer forwards the obsolete owner operation traits.
- Linalg and ordinary FFT eager operations use `EagerSessionLinalgExt` / `EagerSessionFftExt`; integration tests, examples, benchmarks, documentation snippets and the shipped compute-skill mirrors use borrowed routes. `EagerTensorLinalgExt::solve` remains tensor-owned to preserve the **calling-thread** `no_grad` behavior. Consuming in-place FFT retains its ownership contract. Legacy native-context/owned-only extension execution requires a separate top-level region when a borrowed session reports typed unsupported; no nested fallback was added.
- Callback-local `no_grad` / `capture_trace` guards must be installed inside a CPU session callback: a guard on the caller thread does not cross the worker-thread boundary. CUDA's `reduce_sum_squares_read` accepts resident views via `read_input`. The CPU, CUDA and WebGPU session-entry/lifetime contracts and typed errors remain covered by focused tests.

## Retained artifact revisions (not new results)

- Repository-local `docs/testing/session-route-baseline.json`: baseline `25da8d431`, original harness `b16c8ce3f`; `docs/testing/session-route-baseline-recaptured.json`: same baseline, recapture harness `5c55496fe` (96 cases). The comparison tooling is in `scripts/`; neither baseline certifies the final eager migration.
- Separate local benchmark worktrees: `tenferro-benchmark` #107 (`bench/1929-session-public-route`, `c08c859`) retains `cpu/small_work` and existing public API, CPU ops/einsum, linalg JVP/VJP, permutation and GPU suites; `strided-rs-benchmark-suite` #41 (`bench/41-quick-full`, `6c647c0`) retains `kernel_scaling` quick/full profiles, `quick.v1.json` and its fail-closed gate. These branches are **locally committed only**, not pushed or proposed upstream. Quick/diagnostic or noisy runs are not publication evidence; the migrated Householder small-append lane has **not** been remeasured.

## Verification and limits

- CPU: workspace `cargo check -j 16 --workspace --all-targets`; AD 128 library tests, 354 integration tests plus the separately passed trybuild group (`RUSTC_WRAPPER=` for diagnostic-path matching), and 186 AD doctests plus one compile-fail case; einsum autodiff integration 41, linalg autodiff integration 297; FFT autodiff operation target 35 tests and 27 doctests; linalg autodiff doctests 198. The changed tutorial binaries run. These counts are local targets, not the hosted CI matrix.
- Feature checks: `tenferro-ad`, `tenferro-linalg` and `tenferro-fft` all-targets with `autodiff,cuda,webgpu` pass; existing CUDA FFI warnings remain. The combined optional-feature check including einsum's WebGPU integration tests is **not** claimed: three unchanged `download_webgpu_tensor(&runtime, out.to_tensor().unwrap())` calls pass a `Tensor` rather than `&Tensor`.
- A100 CUDA: focused borrowed reduction residency and tracked-gradient check, dot-general VJP, eager backward, linalg/FFT device checks and resident-view norm/QR/Householder cases pass. Complete-pivot LU inside a borrowed CUDA region reports typed unsupported (requires a separate top-level native-context region); CUDA pseudoinverse's Boolean comparison intermediate is typed unsupported. These are capability limits, not device-success claims.
- WebGPU: the focused supported-F32 exact-target and borrowed-copy tests pass; `webgpu_add` returns typed unsupported, so the test makes no arithmetic-support claim.
- Documentation snippets, documentation consistency, agent-skill mirror check, operation-category strict check, session-entry audit and `git diff --check` pass. Hosted CI owns the broader matrix.

## Deferred, with reasons

- **A2** (`with_evaluation_scope` and AD wiring) is deferred by the revised contract to #1938 / #1927 / #1904. The existing eager backend is a non-`Clone` composite; its `!Send` mutex guard cannot cross `CpuOperationEntry::enter`'s `Send` callback, while taking an evaluation scope before that mutex reverses the permit/lock order. A concrete same-domain handle and ownership decision are needed. The audit's zero-entry A2 allowlist prevents accidental introduction.
- **CPU entry floor and eager small-op residue:** #1904/#1927 own the single-worker entry cost. The bench-only scope intervention was reverted; its measured ~2× linalg VJP and −14…−18% eager small-op changes are **not** delivered improvements. No quiet-host, paired, full-manifest or noise-band campaign was required or run.
- **GPU performance/lifetime work:** #1928 / #1925 / #1924 / #1887 own device lifetime, round-trip and scratch optimizations. Moving the operation bodies to sessions does not constitute a fresh GPU performance result.

## Local gate and handoff state

`bash scripts/check-pr-fast.sh --no-fetch --coverage-reviewed --test 'cargo test -j 16 -p tenferro-ad --test integration eager_reductions_and_reverse_validate_axes_before_ad_recording'` passed (formatting, documentation snippets, session-entry audit, root and standalone-manifest clippy, focused test). A deterministic repository-rules **worktree preview** also passed; review must be repeated on the committed HEAD before pushing. The self-reviewed source and handoff are local until the branch is committed and pushed; no PR is created as part of #1929.
