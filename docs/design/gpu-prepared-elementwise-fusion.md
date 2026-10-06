# Execution path parity and prepared-path elementwise fusion

Status: agreed design for the elementwise region work. Stage 0 is implemented as
tests; the production change (Stage 1) is not started. Reviewed by a multi-model
council; the review changed the shape of the plan (Stage 2 is a non-goal for
now, dependency handling is not a union, multi-output regions stay supported).

## Problem and evidence

The production executor had no elementwise fusion.

- The scheduled executor (`execute_scheduled_reads` -> `execute_scheduled_slots`)
  is the only production executor: `run_compiled`, `run_compiled_values`,
  `run_prepared`, and scoped/admitted execution all reach it. It dispatched one
  command per scheduled instruction.
- The legacy segmented executor (`segment.rs`, reached through the erased
  executor's whole-program `execute*` entry points) does fuse, but a caller scan
  shows every one of those entry points is invoked only from the `mod tests`
  module in `runtime/execution.rs`. It is production-dead code, which is why the
  divergence was invisible: the fusion existed only where nothing ran it.
- `gpu/elementwise` measures the consequence directly: a pure elementwise chain
  costs one command per instruction in production.

Measured divergence: a pure elementwise chain `y = tanh(exp((x+x) * (x+x)))`
compiles to four elementwise instructions; the prepared path submits four
commands, while `segment_exec_program` groups them into one fusable segment
(census: `fused_segments=1(4 instr)`, `fusable=1(4 instr)`, `consecutive=1`).

Benchmark census of prepared programs: dense/einsum problems are a single FFI
extension op (nothing to fuse); `gpu/tensornetwork` is 549 FFI einsum extension
ops whose `broadcast_multiply` work happens inside tenferro-einsum
(`outer_product`, already one fused command per node); linalg AD
`grad_sum_qr_jvp` 256 has 25 instructions (1 host / 5 ffi / 19 other) forming 5
segments that fusion eligibility rejects because the runs contain `ReduceSum`
and `Transpose`. So this change does not reduce the current benchmark kernel
counts; it removes a general divergence and the per-command cost of elementwise
chains.

Cost context: in the `gpu/tensornetwork` trace profile consecutive launch
submissions are a median 54.1 us apart while the driver launch call itself is
4.1-10.8 us, so each avoidable command is worth tens of microseconds.

## Stage 0 (implemented)

Two tests, no production change:

1. Numeric parity: the same chain through `run_compiled` and through
   `prepare_compiled` + `run_prepared` must produce identical results.
2. Plan census: the prepared program for the chain holds the elementwise
   instructions and `segment_exec_program` yields exactly one fusable segment
   covering all of them, which is the structural statement of the divergence.

Submission counts are not asserted yet; the harness for counting real
submissions per path is a follow-up (the council asked to keep results, plan
census, and submissions distinct).

## Stage 1 (to implement): prepared executes regions

Grouping is decided at prepare time, after placement and dependency resolution,
and the outer schedule topology and boundary contract do not change at run time.

- Reuse the existing segmentation as *candidate* extraction; `Segment::Fused` is
  a candidate classification, not a proof of fusability. Apply one shared
  eligibility check so both paths reach the same conclusion for the same inputs.
- Eligibility: pure elementwise instructions inside one executor, location and
  event domain; never crossing Host, FFI, Transfer, or Barrier.
- Region I/O: enumerate external inputs and **all live-outs** (future uses and
  graph outputs included). Multiple outputs are allowed where the existing
  builder and backend already support them; the earlier single-exit restriction
  is withdrawn. Split at a boundary that keeps unsupported values live; never
  drop an intermediate silently.
- Dependencies: keep the instruction-id to region-node mapping, remove internal
  dependencies, and reconnect external ones. A union of member dependencies is
  not sufficient; the reduced graph must stay a DAG with the required ordering.
- Completion: one region completion must cover every submission the region
  makes, in the fused and the fallback case. Sharing a domain is not by itself a
  witness.
- Slots and lifetimes: stage external inputs and validate every live-out;
  non-materialized intermediate slots are not validated. Derive retention from
  region-boundary liveness instead of reusing per-instruction `last_use`.
  Fallback intermediates, aliased backing storage, and borrowed inputs stay
  alive until their last GPU use. Logical slots may disappear; physical
  allocations may not be released or reused early, on success, mid-run error,
  event-record failure, or unwind.
- Fallback and dispatch: fall back to per-instruction execution only when no
  submission has happened and the input and slot state can be reused safely; do
  not re-execute after a partial-submit error. Reference
  `PreparedOperationPlan` by original instruction id, never by schedule index.
  Do not store run-specific handles or completion tokens in cached plans.

## Stage 1 status

Implemented for tensor output mode: a planned region's start node executes the
whole region as one fused command through the new
`ErasedTensorBackendExecutor::execute_elementwise_fusion_slots`, and every
covered node's completion resolves to the region's completion token, which the
event domain records after the region's last command. When the backend declines
the fusion the region dispatches its instructions one by one inside the same
enqueue, so one region completion still covers every submission. Every live-out
is validated for presence and residency before it is published, and region
inputs whose last use is inside the region are released at the region boundary.

Regions apply only in tensor output mode; value mode keeps the per-instruction
path, which preserves its contract. The fused entry point takes owned tensors, so
a region with a borrowed view input materializes each view once
(`VIEW_COPY_MIN_INSTRUCTIONS = 3`) and fuses; below that length the copy cannot be
amortized and the region keeps the per-instruction path, which resolves views
itself. This is what makes a chain over graph inputs or embedded constants fuse
on CUDA: without it the chain fell back to one command per instruction
(`gpu/elementwise` trace 0.975 ms at n=1048576 against 0.190 ms after).

An engine whose fused kernel accepts a bounded number of inputs reports that
bound through `ElementwiseRuntime::max_fused_region_inputs`. The planner skips a
candidate region that reads more distinct external inputs than the root engine's
bound, so the region runs instruction by instruction instead of attempting a
fusion that the backend declines on every execution (#1968). The CPU engine
reports the erased strided fused kernel's limit
(`tenferro_cpu_fused::ERASED_FUSION_MAX_INPUTS`, four inputs); CUDA reports no
limit, so its region planning is unchanged. The CPU fused entry point also
declines such plans itself, which keeps the unprepared segmented path correct.
`prepared_elementwise_region_above_cpu_input_limit_is_not_planned` guards the
planner side.

Re-divergence is guarded from two sides. The plan census
(`execution_command_counts`, computed in `region.rs` from the shared
segmentation and eligibility) is asserted against the segmented executor's count
in the parity matrix, and `tests/integration/execution_path_contract.rs` scans
the runtime sources so that segmentation, eligibility, and region construction
each have exactly one definition site and the executor cannot decide eligibility
itself.

Runtime evidence: `PreparedCompiledGraph::elementwise_region_execution_counts`
records fused and fallback executions.
`prepared_elementwise_region_executes_as_one_fused_command` asserts one fused
execution and matching results against the unprepared path for a chain above the
CPU fused kernel's element-size floor, and
`prepared_elementwise_region_falls_back_when_fusion_is_declined` asserts a
single fallback and matching results below that floor.

Two findings from this slice:

- `run_compiled` and `run_prepared` share one prepared program through the
  runtime's prepared-entry cache, so their execution counters and any prepared
  state are already shared. Tests must isolate the counted runtime to observe one
  path's execution.
- Materializing a borrowed view input for a region is now implemented with an
  explicit amortization floor (`VIEW_COPY_MIN_INSTRUCTIONS = 3`); a short region
  with a view input still declines the fusion.

## Stage 2 decision (recorded)

There is no second production executor to unify or delete: `run_compiled*`
already prepares through the runtime's prepared-entry cache and then runs the
scheduled executor, so it differs from `run_prepared` only in re-preparing (a
cache lookup) instead of taking a caller-held handle. Making `run_compiled*` a
wrapper over prepare+run is therefore a refactor of the entry points, not a
behavioral unification, and it is not needed for parity.

What remains is dead code: the segmented executor and its `execute*` entry
points exist only for tests and mislead readers (this investigation was misled by
them). `execution_path_contract.rs` now asserts those entry points are called
only from the test module, so they cannot quietly become a second production
path. Deleting the dead segmented executor is a separate cleanup change; it is
not part of this one.

## Acceptance criteria

- The known four-instruction chain submits exactly one fused command in both
  paths (not merely "the same count").
- Forced fallback dispatches the original four instructions and matches an
  independent oracle.
- Numeric comparison uses the existing per-dtype contract; command count alone
  never implies a new bitwise requirement.
- Multiple live-outs, duplicate returns, and both Tensor and Value output modes
  keep the outputs they need.
- Fusion -> FFI -> Fusion, mixed locations, and ineligible operations keep
  correct dependencies, dispatch, and splitting.
- Delayed completion, mid-fallback failure, and event-record failure never
  reclaim an allocation early, and repeated runs do not accumulate completed
  transient resources.
- Runtime id/epoch and input-signature mismatches keep their current behavior.
- If shared code changes, CPU and any available non-CUDA backend regressions are
  checked; anything not run is reported as not run.
- Existing GPU tests gain no new failures or unsupported results.
- Performance claims record the effective single-worker/thread configuration and
  rest on at least three repetitions with the spread shown.
- Design contracts land in `docs/design`, decisions and verification in a
  worklog.

## Verification layers

- Structure: region formation, all live-outs, dependency reconnection, DAG
  preservation, boundary splitting.
- Numeric and dispatch: independent oracle, broadcast and view inputs, duplicate
  slots, terminal outputs, multiple outputs, FFI interleaving.
- Async: delayed tokens, partial submits, event-record failure, allocator reuse,
  cleanup on error and unwind.
- Real CUDA: one fused submission for an eligible chain, checked separately from
  the kernel launches it produces.

Benchmark: `gpu/elementwise` (pure elementwise chain, one large bandwidth-bound
size and one small per-command size, participants tenferro-cuda-trace,
tenferro-cuda-eager, pytorch-cuda) plus a permanent command-count test, since the
benchmark alone cannot show which path submitted what.

## Open questions to settle before implementation

1. **Answered (code reading).** `collect_segment_inputs` borrows
   (`Vec<&Tensor>`) and inputs are reclaimed only after a successful fusion
   (`reclaim_segment_inputs_exec` on `last_use` slots), so a fusion that returns
   `None` leaves the inputs and slots untouched and the fallback can run. The
   remaining assumption is that a backend's `execute_elementwise_fusion` returns
   `None` without partially writing outputs; the CUDA and CPU implementations
   must be checked for that, and the prepared region executor must not reuse the
   segmented `last_use` reclamation, which the design already forbids.
2. **Answered (code reading).** The CUDA event domain arms a
   `SubmissionCleanupGuard` before running the launch closure. On a launch
   failure or an event-record failure it synchronizes the stream
   (`finish_with_error` -> `synchronize_stream`) before returning the error, and
   on unwind its `Drop` retires the stream best-effort. So the rule already is:
   nothing is released without a completion witness, and on failure the stream
   barrier is the witness of last resort. Stage 1 must keep that contract for a
   region (fused or fallback), and must not re-execute after a partial submit.
3. **Answered (code reading).** The unprepared path has no event domains at all
   (`segment.rs` has no reference to them), so "one region completion covers
   every submission" is a prepared-path property: the prepared executor records
   the completion event inside the enqueue closure, so it must record it after
   the region's last command in both the fused and the fallback case. This is a
   constraint on the Stage 1 implementation, not an existing behavior.
4. **Answered (code reading).** `build_elementwise_fusion_plan` maps every
   `output_slots` entry to a plan value and stores the list in the
   `ElementwiseFusionPlan`, and the CUDA `fusion::launch` allocates one output
   per `plan.outputs()` and returns `Vec<TypedTensor<T>>`, so multiple live-outs
   are supported end to end. There is no `execute_elementwise_fusion_value`:
   Value mode is handled at the segment level by
   `execute_fused_value_segment`, which uses the Tensor-returning fusion and
   keeps `terminal_slots` handling around it. Stage 1 must therefore keep
   terminal and value handling at the region level and write every plan output
   back to its slot. Residual: a Stage 1 test must cover a terminal output inside
   a fused region in Value mode, using `execute_fused_value_segment` as the
   reference behavior.
5. **Answered (code reading and a new test).** Preparation keys a prepared
   program by `PreparedEntryKey` = (root, requirements, specialization), and the
   specialization is projected from the input signature, so a different
   signature cannot reuse a stale prepared program. At run time
   `run_prepared` rejects inputs outside that signature:
   `runtime_prepared_rejects_inputs_outside_its_signature` asserts that an f32
   input for an f64 signature and a length-3 input for a length-2 signature both
   fail.

## Non-goals

Adding `ReduceSum`/`Transpose` fusion, tenferro-einsum's internal outer product,
cubecl per-command overhead, AD optimization, CUDA Graphs, new public API, and a
general IR framework. The broadcast-multiply pair/triplet paths and the
single-session strategy must not be deleted or silently slowed down.
