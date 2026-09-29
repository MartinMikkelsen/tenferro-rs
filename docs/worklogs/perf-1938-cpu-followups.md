# CPU and GPU follow-ups on the #1938 session redesign

Covers #1897, #1898, #1899, #1904 (in-session costs only) and #1884, stacked
on the #1944 session phase. The durable `Auto` rules live in D9 of
[the session redesign](../design/tensor-session-redesign-1938.md).

## Decisions

- **Canonical-fallback GEMM (#1897).** The fallback now builds its GEMM plan
  straight from the canonical axis order instead of re-running the general
  analysis on packed operands, and the owned-layout validation takes a
  storage-length fast path. Allocations per call dropped from 25 to 15; the
  remaining ones are in strided-rs (views hold `Arc` dims, and the HPTT plan is
  rebuilt on every call). Those need a strided-rs change and a pin bump, so they
  are out of scope here.
- **Rejected: a packed operand as pooled scratch read through a view.** It saved
  only 4.3 → 4.2 µs, within noise, and needed four new `PooledUninitOutput`
  methods. Reverted.
- **`Auto` inside sessions (#1898), per the maintainer's choice to parallelize
  both strided and grouped batches.**
  - The fan-out is gated by a per-item cost model (about 50 ns plus m·n·k/16 ns
    per item, at least 8 µs of work per lane). A pure multiply-add threshold
    either missed large batches of tiny items, whose cost is per-call overhead,
    or made 16 threads slower than 1.
  - Grouped lanes run one contiguous job chunk per provider call. One call per
    job cost about 0.4 µs of request setup, against 0.1 µs per job inside one
    call, and made 1024 jobs of 4³ four times slower at 4T.
  - Chunk disjointness comes from increasing output starts plus the validator's
    pairwise disjointness. Explicit per-chunk unions were rejected because their
    serial O(jobs) setup cost as much as the tiny GEMMs themselves.
- **Small GEMMs stay on faer `Par::Seq` below 2^20 real multiply-adds** (complex
  elements count as 4), so a multi-threaded backend is not slower than one
  thread for small products. The threshold comes from a single-GEMM sweep:
  parallel lost at 64³ and won at 128³.
- **#1904 session entry is out of scope, by maintainer decision.** Sessions are
  closure-scoped because faer's `Par::rayon` runs on the ambient pool. An inline
  single-thread path would complicate the parallel model to serve small FFI
  calls. The explicit-executor design is issue #1945. In-session tiny-GEMM
  overhead is fixed here.
- **#1899.** `ConcreteEinsumPlan` already reused its prepared binary-dot plan for
  `execute_into`. The accumulating variant now does too. A swapped operand order
  swaps the conjugation flags, which name the einsum inputs. The prepared path
  is still about 0.1 µs above the plain call because of its own validation. The
  gain beyond that needs a prepared GEMM-analysis cache slot, and that contract
  is a design decision left open.
- **#1884.** The faer half is fixed by the batch lanes. The LAPACK packed-LU loop
  used to ignore the batch policy. It now accepts only `Auto` and
  `ProviderItems` and rejects the other strategies with a typed error. The BLAS
  half, provider-threaded `?getrf` per small matrix, still needs an engine-level
  scheduling decision.

- **GPU (#1923, #1924, #1925).**
  - `cusolverDnXsyevBatched` now takes computeType equal to the type of A.
  - Each CUDA vendor library (cuTENSOR, cuSOLVER, cuBLAS, cuFFT) is loaded
    once per process. A failed load is not cached.
  - Borrowed read views reuse the buffer's memoized device address, as owned
    operands already did.
  - Borrowed destinations keep the blocking `get_resource`. It orders the
    vendor write after CubeCL work that is still queued and may read the
    destination. The owned-destination fast path skips that ordering: a vendor
    write through a memoized address could overtake a queued CubeCL read of
    the same buffer. That write-after-read window predates this change and is
    left for a separate decision.

## Verification conclusions and constraints

- Value tests cover:
  - borrowed operands at nonzero offsets and padded outputs in the canonical
    fallback;
  - padded batch strides across the strided lane split;
  - ordered and reversed grouped jobs, taking the chunk path and the per-job
    path;
  - the conjugation swap on the swapped accumulate path.

  Each new guard was checked to fail with its fix removed.
- Timings come from a shared EPYC host with a load average around 5–7. Absolute
  numbers are noisy, and short-sample rows are bimodal. Only the direction and
  rough size of each change are claimed.
- Headline numbers (1T → 4T, or before → after):
  - #1898 batch 4×4×4, b = 1024: 72 → 30–33 µs.
  - Grouped 64³ × 8: 108 → 51 µs.
  - Single 32³ at 16T: 28 → 3.6 µs.
  - #1897 canonical: 5.5 → 4.3 µs.
- BLAS batched-LU timings were not re-measured. No local OpenBLAS here provides
  both LAPACK and `cblas_?gemm_batch` for a release probe.
- Regression cases for these fixes are added to tenferro-benchmark separately.
- GPU measurements come from a local A100:
  - Vendor-library churn: before, RSS grew 410 → 3114 MB in 50 create/drop
    cycles; after, it stays flat at about 662 MB over 300 cycles.
  - In-session view-operand GEMMs: 36.6 → 19.7 µs at 8×8, against 18.5 µs for
    owned operands.
  - The local GPU suites pass, apart from two things. One source-contract
    needle was updated for the refactor. Nine tenferro-linalg doctests
    (`shape() == &[]`) fail to infer types under `--features cuda`; they are
    untouched by this branch, and hosted CI does not run cuda doctests.
