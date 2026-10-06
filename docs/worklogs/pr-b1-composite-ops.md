# #2010 PR-B1: `erf` and composite activation, softmax, normalization and `take_along_axis` ops

## Decisions
- `erf` is a core primitive (D-6), real `F32`/`F64` only, wired like `log1p`.
  The CPU scalar comes from `libm` because `f64::erf` is unstable in std and no
  existing workspace dependency provides it. CPU fusion declines regions
  containing it (`strided_fused` has no instruction); a strided-rs `FusedOp::Erf`
  is a later upstream item. StableHLO lowering rejects it explicitly instead of
  emitting CHLO, which the emitter does not produce today.
- Every composite is written once over a small primitive vocabulary
  (`CompositeOps` in `tenferro-runtime/src/composite.rs`) that the traced,
  eager and concrete-session surfaces implement. Rejected: three hand-written
  copies per op, because the edge-case policy (all-masked softmax, `eps`,
  empty axes) would drift between surfaces and backends. The module is
  `#[doc(hidden)] pub` only so `tenferro-ad` can implement the trait for its
  eager session, following the `scale` precedent.
- Activations use `select(x > 0, -x, x)` for `-|x|`, not `-abs(x)`, so AD gives
  the exact derivatives at 0 (`softplus'(0) = 1/2`, `softplus''(0) = 1/4`,
  `sigmoid'(0) = 1/4`). All intermediates and derivatives stay finite for
  finite and infinite inputs, except where the documented product
  `-inf * 0` gives `NaN` (`silu`, `gelu` at `-inf`, matching PyTorch).
- Softmax handles an all-masked slice by clamping the slice maximum to the
  lowest finite value and the slice sum to the smallest positive normal, with
  a NaN-propagating `maximum`. The value is then `0` / `-inf`, the gradient
  stays finite (it reaches the clamp constants, not `x`), and a `NaN` maximum
  still propagates. The first version used `compare(max == -inf)` and two
  `select`s, which cost three extra small ops and three constants per call.
  The masked forms' `select(mask, x, -inf)` makes masked-out gradients exactly
  zero.
- Composites emit no more ops than the corresponding hand composition. Scalar
  constants are broadcast once to the shape they are used at (`splat`), never
  per binary op. The normalizations use the fused `reduce_sum_squares` and
  `scale_real`, exactly like the hand composition in the
  tenferro-benchmark `cpu/perf_issues` #2006 case. The first version spent an
  extra full-size `mul` and per-op scalar broadcasts, which made the
  single-call API slower than the hand composition (follow-up PR to #2017).
- `take_along_axis` builds `(index, batch coordinates)` tuples on the backend
  from scalar constants with a log-depth doubling `iota`, one-hot `pad` vectors
  and broadcasting arithmetic. Host-built coordinate tensors were rejected:
  a traced graph cannot carry shaped constants, and an attached host input has
  no ingress on a device runtime. `concatenate` is used only between constants,
  because concatenating distinct traced inputs fails at compile without input
  specs (a separate, pre-existing traced bug, #2018).
- CUDA `to_contiguous_read` now materializes `Bool` tensors and strided views
  (bool copy kernel, `u8` native permutation). Eager AD retains `select`
  conditions through it, so tracked `where_select` on CUDA failed before.

## Verification conclusions and constraints
- CPU `erf` is within 1 ulp of an mpmath (200-bit) reference over the core
  interval, both tails, saturation, subnormals and the IEEE specials; CUDA is
  within 4 ulp of the CPU kernel, and exact on the specials.
- Values, first and second derivatives, the edge-case policy and
  finite-difference gradients are tested on the eager, traced and
  concrete-session surfaces. CPU/CUDA parity of values and gradients is tested
  on an A100 for every composite, F32 and F64, including traced fused regions
  containing `erf`.
- The traced composites require concrete input shapes.
- The fused CPU `layer_norm` / `rms_norm` kernel, `argmax`, index views
  (`TensorRead`) for `take_along_axis`, and a StableHLO `erf` lowering remain
  out of scope (B2 / follow-ups).
