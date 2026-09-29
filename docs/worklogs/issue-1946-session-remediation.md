# #1946 session, representation and host-family remediation

Batched remediation of the defects found after #1938 / #1944, delivered as one
non-squash tenferro-rs PR plus one tenferro-benchmark PR (the benchmark PR
merges after this one). Durable contracts live in
[`explicit-session-boundary.md`](../design/explicit-session-boundary.md),
[`tensor-session-redesign-1938.md`](../design/tensor-session-redesign-1938.md)
and the "Backend Session Entry" section of `REPOSITORY_RULES.md`.

## Decisions

- **No self-entering operation APIs.** A public operation borrows a session; it
  never opens one. Eager einsum/tensordot moved to `EagerSessionEinsumExt`
  with no compatibility shim (F5). Tensor-owned `solve` (calling-thread
  `no_grad`) and consuming in-place FFT (exclusive ownership) remain the two
  reviewed exceptions.
- **Nested entry fails fast (F1).** A thread holding a session, a portable
  session or a CPU permit is rejected before it would wait on an eager owner
  lock; inside a shared execution scope a busy owner is `Contended`. A blocking
  wait was the deadlock.
- **Lane policy is data (F2/F3).** The allocating batched dot takes the same
  lane decision as the caller-owned path, and the thresholds live in
  `CpuBatchThresholds` instead of constants.
- **Owner types are session hosts (F6).** Owner impls of canonicalization,
  fusion and buffer traits entered a session behind the caller; they were
  removed rather than documented. `TensorDeviceTransfer` stays on owners: it is
  the explicit transfer boundary and enters no session. The owner-only BLAS
  linalg mode helper went with the owner entry points; production linalg
  already used the session's engine mode (see constraints).
- **Representation markers survive metadata transforms (F7)**, and
  metadata-only typed view transforms use the `_view` suffix (F10); `try_` is
  kept for variants whose error returns the input.
- **One host tensor family (F8).** `HostTensor`/`HostTensorView` were deleted,
  not aliased: scalar-set members, `DefaultScalars` and `ErasedHostTensor`
  carry `TypedTensor<T, DynRank, Host>`, and `ErasedHostTensor` presents a
  canonical `TensorLayout`. `DefaultScalars` keeps its shape (it is tenferro's
  own `ScalarSet`); folding it into `Tensor` would have touched backend
  variants and was rejected as redesign. Set value enums keep `Clone` (host
  typed tensors are `Clone`) and drop `PartialEq`, which nothing used.
- **The session-entry audit runs in hosted CI** and rejects `PENDING`
  allowlist reasons, so a known-illegitimate entry cannot merge.

## Verification conclusions and constraints

- Each F item has a focused regression test that fails on the pre-fix code
  (F1 watchdog tests, F2 route parity, F3 threshold boundaries, F4 trace
  capture, F7 compile-level marker annotations).
- The removed owner BLAS linalg mode differs from the session engine mode only
  for a BLAS backend on an executor without Rayon inner parallelism, where
  linalg runs sequentially instead of with BLAS inner threading. This predates
  #1946 on `main` and is recorded, not re-tuned.
- Pre-existing and out of scope: several linalg doctests fail under
  `--features cuda`; the memo fast path has an owner-destination WAR hazard;
  `to_contiguous_read`'s default rejects strided views.
