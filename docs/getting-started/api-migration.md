# API migration guide

This page is the first stop when an older tenferro-rs example fails with
`cannot find module`, `cannot find function`, or a changed-signature error.
The current public API favors explicit extension traits, fallible constructors,
and session-owned execution. Historical worklogs are not API documentation.

## Removed modules and free functions

| Older spelling | Current spelling |
| --- | --- |
| <code>tenferro_einsum::&#8203;eager_tensor</code> | Import `tenferro_einsum::EagerSessionEinsumExt` and call the trait method on a borrowed session: `ctx.with_eager_session(\|s\| s.einsum(&[&a, &b], "ij,jk->ik"))??`. |
| `tenferro_einsum::EagerEinsumExt` (`[&a, &b].einsum(...)` on eager values), `EagerTensorEinsumExt` (`a.tensordot(&b, axes)`) | `EagerSessionEinsumExt` on a borrowed session: `s.einsum(&[&a, &b], "ij,jk->ik")`, `s.einsum_notation(...)`, `s.einsum_subscripts(...)`, `s.tensordot(&a, &b, axes)`. |
| <code>tenferro_linalg::&#8203;eager_tensor</code> | Import `tenferro_linalg::EagerTensorLinalgExt` and call the trait method on the eager value. |
| <code>tenferro_runtime::&#8203;traced_tensor</code> | Use the current traced tensor types and their extension traits, such as `tenferro_linalg::TracedTensorLinalgExt`. |
| `tenferro_einsum::einsum` | Use the current trait method, for example `TraceContextEinsumExt::einsum` or `TracedTensorEinsumExt::einsum`, for the receiver you have. |
| `tenferro_einsum::einsum_subscripts_with` | Import the owning einsum extension trait and call its session/context method. |
| `tenferro_linalg::svd`, `qr`, `eigh`, `solve` | Import `TensorLinalgExt`, `TypedTensorLinalgExt`, or `TensorReadLinalgExt` and call `.svd(...)`, `.qr(...)`, `.eigh(...)`, or `.solve(...)` on the input. |
| `tenferro_tensor::HostTensor<T>` / `HostTensorView<'a, T>` | `TypedTensor<T, DynRank, Host>` (`from_host_vec_col_major`, `as_slice`, `host_data_mut`) and `TypedTensorView<'a, T, DynRank, Host>` (`from_host_slice`, `as_host_slice`). Scalar-set members and `ErasedHostTensor` payloads use the same types. |
| `tenferro_tensor::core::{DefaultScalars, HostTensor, ...}` | `tenferro_tensor::core` re-exports metadata only; import tensor types, including `DefaultScalars`, from the `tenferro_tensor` root. |
| `TypedTensorView::try_slice`, `try_slice_axis`, `try_reshape` (and the `TypedTensorViewMut` forms) | `slice_view`, `slice_axis_view`, `reshape_view`. |
| Owner-side execution on a backend: `CpuBackend::to_contiguous` / `copy_into` (`TensorViewCanonicalization`), `TensorFusion`, `reclaim_buffer`, `with_linalg_pool`, and the CUDA/WebGPU owner equivalents | Enter once and call the method on the session: `backend.with_backend_session(\|s\| s.to_contiguous_read(read))??`. |

The owning crate's `prelude` re-exports the public operation traits. For a
first direct CPU program, the usual imports are:

```rust
use tenferro_cpu::CpuBackend;
use tenferro_linalg::prelude::*;
use tenferro_runtime::prelude::*;
```

Then keep execution inside one session:

```rust
let mut backend = CpuBackend::new();
// The outer `?` reports session admission; the inner one the operation.
let values = backend.with_backend_session(|session| input.svdvals(session))??;
```

## Autodiff context changes

`AdContextBuilder::with_core_rules()` was removed. The core primitive rules
are installed by the normal builder; start with:

```rust
let ad = tenferro_ad::AdContext::builder().build()?;
```

When an operation family supplies semantic AD rules, install that family
explicitly instead:

```rust
let ad = tenferro_ad::AdContext::builder()
    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
    .build()?;
```

## Fallible constructors and shape APIs

Constructors that validate storage, placement, or metadata return `Result`.
Propagate the result instead of relying on an infallible constructor:

| Older assumption | Current spelling |
| --- | --- |
| `EagerTensor::from_tensor_in(tensor, ctx)` returns a value | `EagerTensor::from_tensor_in(tensor, ctx)?` |
| `EagerTensor::requires_grad_in(tensor, ctx)` returns a value | `EagerTensor::requires_grad_in(tensor, ctx)?` |
| `TracedTensor::input_concrete_shape(dtype, shape)` is infallible | `TracedTensor::input_concrete_shape(dtype, shape)?` |
| `TypedTensor::from_vec_col_major(shape, data)` is infallible | `TypedTensor::from_vec_col_major(shape, data)?` |
| `TypedTensor::zeros(shape)` is infallible | `TypedTensor::zeros(shape)?` |

`reduce_sum` uses an explicit optional axis list on eager values:

```rust
let total = value.reduce_sum(None)?;          // all axes
let columns = value.reduce_sum(Some(&[0]))?;  // selected axes
```

An empty slice remains distinct from `None`: use `Some(&[])` when the API's
identity/no-axis behavior is what the program needs.

## Finding the current method

1. Choose the value tier: direct concrete tensor, eager tensor, or traced tensor.
2. Import the `*Ext` trait owned by the operation crate.
3. Check the method's receiver and session arity in the
   [API cheatsheet](https://tensor4all.org/tenferro-rs/skill-references/api-cheatsheet.md).
4. Use the [linear algebra guide](../guides/linear-algebra.md),
   [einsum guide](../guides/einsum.md), or
   [custom operations guide](../guides/custom-operations.md) for the relevant
   workflow.

Do not add a compatibility alias for a removed API. If a current example or
error message contradicts this page, report the documentation gap through the
[issue-intake procedure](https://github.com/tensor4all/tenferro-rs/blob/main/ai/contribution-workflows/issue-intake.md)
after obtaining maintainer/user approval.
