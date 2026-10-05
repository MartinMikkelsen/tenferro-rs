# CPU linalg: every family through the batched tlinalg providers (#1956)

## Decisions
- The remaining CPU families (Cholesky, triangular solve, LU and solve including the
  direct-output routes, full-pivot LU and its solve, QR, rank-revealing QR, eigh, eig and the
  compact Householder kernels) moved into `tlinalg` (faer) and `tlinalg-blas` (LAPACK/BLAS),
  next to SVD and packed LU. No numerical kernel or batch loop is left in `tenferro-linalg`.
- The shared `tlinalg-traits` crate is gone: the interface the providers are adapted to is
  tenferro's, each provider owns its vocabulary, and the two adapters map their errors
  separately (`tlinalg-blas` adds an `Internal` variant for impossible LAPACK pivots).
- The provider API is torch-style batched (one call per batch over `[rows, cols, batch...]`).
  The host resolves one lane plan per faer call from the existing batch policy; `tlinalg` owns
  the fan-out on the context's pool. This replaces the host's per-item loops and its packed-LU
  `for_each_chunk`.
- Host-side responsibilities that stay here: tenferro's shape errors, empty results, tensor
  construction, the QR gauges, the rank-revealing QR rank decision (and, on the faer route, the
  screening of non-finite and all-zero items), the compact Householder state compositions, and a
  one-time compact copy of a borrowed operand with negative strides.
- The tprims linalg adapter in `ext/tenferro-cpu-tprims` is removed with its `tprims-linalg`
  dependency; the injectable-kernel slot itself (`cpu_kernels`) is unchanged.

## Verification conclusions and constraints
- Intended behaviour change: non-LU faer families now follow the batch policy, so `Auto` fans
  them out over the context's lanes and an unservable forced strategy is a typed error for them.
  Its timing effect is not measured here; small-matrix batch cases in tenferro-benchmark are the
  follow-up evidence.
- Steady-state allocation ceilings (`tests/cpu_linalg_allocation.rs`) were recorded on the
  pre-move main and lowered where the batched providers allocate less. Four faer cases rose by
  one allocation, each caused inside the provider and marked in the ceiling table: the solve
  destination-aliasing check, the full-pivot solve's lane work matrix, and the rank-revealing
  QR pivot scratch.
