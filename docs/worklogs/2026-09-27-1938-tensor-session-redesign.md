# #1938 tensor/session architecture migration

Target contract: [tensor-session-redesign-1938](../design/tensor-session-redesign-1938.md). The #1929 handoff commit `e55c0e17a` is the implementation baseline; this branch is not an integrated-main performance baseline.

## Decisions

- Begin the output side of the shared storage/dispatch/session seam with an owning pooled full-overwrite lease. It owns a handle to the existing shared pool state, not a mutable borrow of the whole pool during kernel execution. Retain its existing checkout accounting and weak return target; do not introduce another allocator or zero uninitialized output.
- Build rank-changing real-view metadata directly into the existing inline-capacity `ShapeVec`/`StrideVec` rather than allocating intermediate `Vec`s. High ranks retain the existing spill behavior.

## Verification conclusions and constraints

- The CPU basic library tests and doctests passed; a CPU erased-output test demonstrates two simultaneous differently typed leases, safe discard of incomplete scratch, and publication after full initialization. The workspace all-targets check passed after these changes.
- Allocation-counted tests establish zero metadata allocations for small static- and dynamic-rank immutable and mutable complex-to-real views. Existing storage reinterpretation suites pass for both rank modes, including a high-rank spill case.
- These results do **not** establish the rest of D1–D12: ordinary tensor owners still carry groups, erased operation inputs still use their original descriptors, and fallible session entry, native-token recovery, GPU ordering, runtime/AD migration, integrated numerical verification and performance assessment remain outstanding. Do not claim an integrated vertical slice or PR readiness from these checks.
- The next vertical slice cannot replace the host constructor alone. `TensorCore<R>` currently stores an `OwnedTensorGroup<R>` and a layout, and `Tensor` erases preset dtypes by borrowing that same core as `#[repr(transparent)] TypedTensor<T, R>`. A group-free `Vec<T>` needs typed storage and an erased representation that retains its ownership and dtype without a conversion back into a group. Change owner/erasure and borrowed operation input together; a Host-only wrapper that reconstructs the old group at dispatch would invalidate the intended result.
