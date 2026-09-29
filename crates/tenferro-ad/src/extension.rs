//! Eager AD support for out-of-tree extension primitives.

use std::sync::Arc;

use computegraph::GraphOperation;
use tenferro_ops::std_tensor_op::StdTensorOp;
use tenferro_runtime::{
    Error, ErrorPhase, ExtensionModule, InputSignature, PrepareCapability, PrepareError,
    PreparedOperationExecutorHandle, Result, Runtime, RuntimeConfigError,
};
use tenferro_tensor::{Tensor, TensorRead, TensorValue};

use crate::eager::{
    eager_capture_active, eager_grad_recording_enabled, record_eager_outputs, EagerRuntime,
    EagerSession, EagerTensor,
};

pub use tenferro_runtime::extension::{
    apply, ExtensionCacheKey, ExtensionCacheLimits, ExtensionCacheSelector, ExtensionCacheStore,
    ExtensionExecutionContext, ExtensionFamilyId, ExtensionOp,
};

/// Closed backend kind selected by the eager runtime owner for an extension.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::{EagerExtensionBackendKind, EagerExtensionTarget};
/// use tenferro_runtime::EngineId;
///
/// let target = EagerExtensionTarget {
///     engine_id: EngineId::new("example.engine")?,
///     backend_kind: EagerExtensionBackendKind::Cpu,
/// };
/// assert!(matches!(
///     target.backend_kind,
///     EagerExtensionBackendKind::Cpu
/// ));
/// assert_eq!(target.engine_id.as_str(), "example.engine");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EagerExtensionBackendKind {
    /// The eager runtime owns a CPU backend.
    Cpu,
    /// The eager runtime owns a CUDA backend.
    #[cfg(feature = "cuda")]
    Cuda,
    /// The eager runtime owns a WebGPU backend.
    #[cfg(feature = "webgpu")]
    WebGpu,
}

/// Exact engine target selected by the eager runtime owner.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::{EagerExtensionBackendKind, EagerExtensionTarget};
/// use tenferro_runtime::EngineId;
///
/// let target = EagerExtensionTarget {
///     engine_id: EngineId::new("example.engine")?,
///     backend_kind: EagerExtensionBackendKind::Cpu,
/// };
/// assert_eq!(target.backend_kind, EagerExtensionBackendKind::Cpu);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[doc(hidden)]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EagerExtensionTarget {
    /// Exact runtime engine selected for this eager context.
    pub engine_id: tenferro_runtime::EngineId,
    /// Closed backend kind selected for this eager context.
    pub backend_kind: EagerExtensionBackendKind,
}

#[cfg(test)]
mod tests;

/// Validate recording eligibility before a consuming extension mutation.
/// This does not grant write authority: callers must still consume the input
/// through `EagerTensor::into_value` and obtain an exclusive tensor borrow.
///
/// # Errors
/// Returns `Error::RuntimeState` for gradient-tracked values, legacy AD traces,
/// or active capture, including capture nested inside no_grad. Saved-value
/// ownership must additionally pass the structural into_value check.
/// Input-signature validation, module-factory and installation errors are
/// propagated unchanged.
///
/// # Examples
/// ```
/// use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};
/// use tenferro_ad::extension::prepare_eager_in_place_input;
/// let input = EagerTensor::requires_grad_in(
///     Tensor::from_vec_col_major([1], vec![1.0_f64])?, EagerRuntime::new()?)?;
/// let result = prepare_eager_in_place_input(&input, "example", |_| {
///     panic!("tracked inputs must be rejected before module construction")
/// });
/// assert!(result.is_err());
/// assert_eq!(input.shape(), &[1]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[doc(hidden)]
pub fn prepare_eager_in_place_input(
    input: &EagerTensor,
    family_id: &'static str,
    module_factory: impl FnOnce(EagerExtensionTarget) -> Result<Arc<dyn ExtensionModule>>,
) -> Result<()> {
    if input.requires_grad || input.trace.is_some() || eager_capture_active() {
        return Err(Error::runtime_state(
            "eager in-place",
            ErrorPhase::Execution,
            "in-place execution requires an untracked value outside trace capture",
        ));
    }
    let target = input.ctx.eager_extension_target()?;
    validate_eager_extension_input_signature(&input.ctx, &target, &[input.tensor_read()])?;
    let module = module_factory(target.clone())?;
    input
        .ctx
        .ensure_extension_module_for_engine(module, family_id, &target.engine_id)?;
    Ok(())
}

/// Adopt an untracked eager tensor value produced by this runtime's backend.
///
/// This is a low-level extension contract for eager composite operations that
/// execute through a lifetime-bound backend session and receive a lazy
/// [`TensorValue`] from the backend. The value must have been produced for the
/// same eager runtime; this helper intentionally does not register gradient
/// metadata and must not be used for tracked outputs.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::adopt_untracked_eager_value;
/// use tenferro_ad::EagerRuntime;
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::{Tensor, TensorValue};
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let value = TensorValue::from_tensor(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
/// );
/// let eager = adopt_untracked_eager_value(ctx, value)?;
/// assert_eq!(eager.shape(), &[1]);
/// assert!(!eager.tracks_grad());
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
/// # Errors
///
/// Returns [`Error::RuntimeState`] when the value cannot be registered in the
/// supplied runtime, including an invalid or incompatible retained descriptor.
#[must_use = "the adopted eager tensor carries the runtime value"]
pub fn adopt_untracked_eager_value(
    ctx: Arc<EagerRuntime>,
    value: TensorValue,
) -> Result<EagerTensor> {
    EagerTensor::new_untracked_value_result(ctx, value)
}

/// Apply an extension op to eager AD tensors.
///
/// # Examples
///
/// ```rust
/// use tenferro_ad::extension::apply_eager;
/// use tenferro_ad::{EagerRuntime, EagerTensor};
/// use tenferro_cpu::CpuBackend;
/// use tenferro_tensor::Tensor;
///
/// let ctx = EagerRuntime::with_cpu_backend(CpuBackend::new())?;
/// let x = EagerTensor::from_tensor_in(
///     Tensor::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap(),
///     ctx,
/// ).unwrap();
/// let _ = &x;
/// let _apply = apply_eager;
/// # Ok::<(), tenferro_ad::Error>(())
/// ```
/// # Errors
///
/// Returns `Error::Validation` with `InvalidArgument` when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// `Error::ContextMismatch` when tensors belong to different eager runtimes;
/// backend, extension, and runtime-state failures retain their typed sources.
pub fn apply_eager(op: Arc<dyn ExtensionOp>, inputs: &[&EagerTensor]) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    let std_op = StdTensorOp::Extension(op);
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    // Native immediate path: resolve the extension engine from the runtime
    // snapshot, prepare the op, and execute through the prepared plan's
    // scheduler-session executor when it is session-capable. This skips the
    // SemanticProgram build/compile + run_compiled cost on every call.
    if let Some(outputs) = try_prepared_eager_extension(&ctx, &std_op, &input_reads)? {
        return finish_eager_extension_outputs(ctx, std_op, inputs, outputs, None);
    }
    let outputs = ctx.exec_outputs_read(&std_op, &input_reads)?;
    finish_eager_extension_outputs(ctx, std_op, inputs, outputs, None)
}

/// Execute a prepared eager extension on the caller's borrowed session.
///
/// The exact eager runtime, input signature, and selected extension engine are
/// checked before dispatch. A legacy context-only executor requires a separate
/// top-level call to [`apply_eager`] after this session is released; this
/// function never re-enters the eager backend or silently opens that fallback.
///
/// # Errors
///
/// Returns a typed context mismatch, validation, unsupported capability, or
/// backend/extension error without replacing its source.
pub(crate) fn apply_eager_in_session(
    session: &mut EagerSession<'_>,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    if !Arc::ptr_eq(session.runtime(), &ctx) {
        return Err(Error::ContextMismatch {
            lhs: session.runtime().id(),
            rhs: ctx.id(),
        });
    }
    let std_op = StdTensorOp::Extension(op);
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    let target = ctx.eager_extension_target()?;
    let executor = prepared_eager_extension_executor(&ctx, &target, &std_op, &input_reads)?
        .ok_or_else(|| {
            Error::unsupported(
                "extension::apply_eager_in_session",
                ErrorPhase::Execution,
                "no session-capable prepared extension executor for this signature",
            )
        })?;
    if !executor.supports_session() {
        return Err(Error::unsupported(
            "extension::apply_eager_in_session",
            ErrorPhase::Execution,
            "the native-context executor requires a separate top-level runtime region",
        ));
    }
    let outputs = session.execute_prepared_extension(executor.as_ref(), &input_reads)?;
    finish_eager_extension_outputs(ctx, std_op, inputs, outputs, Some(session))
}

/// Run one extension op through the snapshot-resolved native prepared path.
///
/// Returns `None` (so the caller falls back to the compiled-program path for
/// the exact op and signature) when the eager runtime has no exact extension
/// engine, the engine cannot prepare the op, or the prepared plan has no
/// executor. Context-only executors use the native-context bridge in their own
/// top-level region. AD recording remains with the caller.
fn try_prepared_eager_extension(
    ctx: &EagerRuntime,
    op: &StdTensorOp,
    input_reads: &[TensorRead<'_>],
) -> Result<Option<Vec<Tensor>>> {
    // The recording test backend has no extension engine; its existing
    // top-level route still falls back to compiled execution.
    let Ok(target) = ctx.eager_extension_target() else {
        return Ok(None);
    };
    let Some(executor) = prepared_eager_extension_executor(ctx, &target, op, input_reads)? else {
        return Ok(None);
    };
    if executor.supports_session() {
        // native-session: scheduler-owned session executor.
        let outputs = ctx.with_extension_execution_context(|extension_ctx| {
            let (session, caches) = extension_ctx.parts_mut();
            executor.execute_in_session(session, caches, input_reads)
        })??;
        Ok(Some(outputs))
    } else {
        // native-context: mandatory `execute` bridge over the erased backend
        // context (for out-of-tree prepared ops without a session executor).
        let outputs = ctx.with_extension_erased_context(|erased, caches| {
            executor.execute(erased, caches, input_reads)
        })??;
        Ok(Some(outputs))
    }
}

fn prepared_eager_extension_executor(
    ctx: &EagerRuntime,
    target: &EagerExtensionTarget,
    op: &StdTensorOp,
    input_reads: &[TensorRead<'_>],
) -> Result<Option<PreparedOperationExecutorHandle>> {
    let StdTensorOp::Extension(ext) = op else {
        return Ok(None);
    };
    let signature = InputSignature::from_reads(input_reads).map_err(|source| {
        Error::runtime_state_source("extension::apply_eager", ErrorPhase::Execution, source)
    })?;
    let PrepareCapability::Prepared(plan) =
        ctx.runtime()
            .prepare_extension_immediate(&target.engine_id, ext.as_ref(), &signature)?
    else {
        return Ok(None);
    };
    Ok(plan.executor().cloned())
}

/// Ensure an eager extension module is installed, then apply the op through
/// the single [`apply_eager`] entry.
///
/// This thin wrapper is retained for module-owner eager call sites (linalg,
/// einsum). Forward execution always routes through [`apply_eager`]'s native
/// prepared path; this wrapper only owns the install-ensure step.
///
/// # Errors
///
/// Returns `Error::Validation` with `InvalidArgument` when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// `Error::ContextMismatch` when tensors belong to different eager runtimes;
/// backend, extension, and runtime-state failures retain their typed sources.
#[doc(hidden)]
pub fn apply_eager_with_extension_session(
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module: Arc<dyn ExtensionModule>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    ctx.install_extension_module(module)?;
    apply_eager(op, inputs)
}

/// Apply an eager extension through the owner-selected engine and backend kind.
///
/// This narrow sibling-crate wrapper is used by FFT, whose module factory must
/// follow the eager runtime's exact backend selection. Input, target, and
/// ingress validation always run before `module_factory`; errors returned by
/// the factory are propagated unchanged. The returned module is then passed to
/// the owner-scoped ensure operation. Forward execution always routes through
/// [`apply_eager`]'s native prepared path.
///
/// # Errors
///
/// Returns [`tenferro_runtime::Error::Validation`] with
/// [`tenferro_tensor::ValidationError::InvalidArgument`] when `inputs` is empty
/// or its length differs from the extension's declared input count. Returns
/// [`tenferro_runtime::Error::ContextMismatch`] when tensors belong to
/// different eager runtimes.
///
/// Returns [`tenferro_runtime::Error::RuntimeStateSource`] when the selected
/// engine is missing through
/// [`tenferro_runtime::RuntimeConfigError::MissingEngine`], an input has no
/// ingress through [`tenferro_runtime::PrepareError::NoInputIngress`], or the
/// cold/missing-registration ensure path rejects the module. Errors returned by
/// `module_factory` are propagated unchanged. Session, cache, and output
/// registration failures retain their typed runtime sources.
#[doc(hidden)]
pub fn apply_eager_with_targeted_extension_session(
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module_factory: impl FnOnce(
        EagerExtensionTarget,
    ) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    let target = ctx.eager_extension_target()?;
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    validate_eager_extension_input_signature(&ctx, &target, &input_reads)?;
    let module = module_factory(target.clone())?;
    ctx.ensure_extension_module_for_engine(module, op.family_id(), &target.engine_id)?;
    apply_eager(op, inputs)
}

/// Install the exact eager-owner extension and execute it on a borrowed session.
/// Input and selected-engine ingress checks precede module construction; a
/// context-only executor returns a typed unsupported error rather than nesting
/// an erased-context entry inside the active session.
///
/// # Errors
///
/// Returns typed context, validation, module-installation, unsupported-executor,
/// or backend errors from the corresponding boundary.
#[doc(hidden)]
pub fn apply_eager_with_targeted_extension_in_session(
    session: &mut EagerSession<'_>,
    op: Arc<dyn ExtensionOp>,
    inputs: &[&EagerTensor],
    module_factory: impl FnOnce(
        EagerExtensionTarget,
    ) -> tenferro_runtime::Result<Arc<dyn ExtensionModule>>,
) -> Result<Vec<EagerTensor>> {
    let ctx = validate_eager_extension_inputs(op.as_ref(), inputs)?;
    if !Arc::ptr_eq(session.runtime(), &ctx) {
        return Err(Error::ContextMismatch {
            lhs: session.runtime().id(),
            rhs: ctx.id(),
        });
    }
    let target = ctx.eager_extension_target()?;
    let input_reads: Vec<_> = inputs.iter().map(|tensor| tensor.tensor_read()).collect();
    validate_eager_extension_input_signature(&ctx, &target, &input_reads)?;
    let module = module_factory(target.clone())?;
    ctx.ensure_extension_module_for_engine(module, op.family_id(), &target.engine_id)?;
    apply_eager_in_session(session, op, inputs)
}

pub(crate) fn validate_eager_extension_target(
    runtime: &Runtime,
    target: &EagerExtensionTarget,
) -> Result<()> {
    let snapshot = runtime.snapshot().map_err(|source| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            source,
        )
    })?;
    if snapshot.engine(&target.engine_id).is_none() {
        return Err(Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            RuntimeConfigError::MissingEngine {
                engine_id: target.engine_id.clone(),
            },
        ));
    }
    Ok(())
}

fn validate_eager_extension_input_signature(
    ctx: &EagerRuntime,
    target: &EagerExtensionTarget,
    input_reads: &[TensorRead<'_>],
) -> Result<()> {
    let signature = InputSignature::from_reads(input_reads).map_err(|source| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            source,
        )
    })?;
    let snapshot = ctx.runtime().snapshot().map_err(|source| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            source,
        )
    })?;
    let engine = snapshot.engine(&target.engine_id).ok_or_else(|| {
        Error::runtime_state_source(
            "extension::apply_eager_with_extension_session",
            ErrorPhase::Execution,
            RuntimeConfigError::MissingEngine {
                engine_id: target.engine_id.clone(),
            },
        )
    })?;
    for (input_index, entry) in signature.entries().iter().enumerate() {
        if !engine.accepts_input_signature(entry) {
            return Err(Error::runtime_state_source(
                "extension::apply_eager_with_extension_session",
                ErrorPhase::Execution,
                PrepareError::NoInputIngress {
                    input_index,
                    placement: entry.placement().clone(),
                },
            ));
        }
    }
    Ok(())
}

fn validate_eager_extension_inputs(
    op: &dyn ExtensionOp,
    inputs: &[&EagerTensor],
) -> Result<Arc<EagerRuntime>> {
    let Some(first) = inputs.first() else {
        return Err(Error::invalid_argument(
            "extension::apply_eager",
            ErrorPhase::Execution,
            "inputs",
            "at least one input tensor is required",
        ));
    };
    if inputs.len() != op.input_count() {
        return Err(Error::invalid_argument(
            "extension::apply_eager",
            ErrorPhase::Execution,
            "inputs",
            format!(
                "op family {:?} expects {} inputs, got {}",
                op.family_id(),
                op.input_count(),
                inputs.len()
            ),
        ));
    }

    let ctx = Arc::clone(&first.ctx);
    for tensor in inputs.iter().skip(1) {
        if !first.same_context(tensor) {
            return Err(Error::ContextMismatch {
                lhs: first.ctx_id(),
                rhs: tensor.ctx_id(),
            });
        }
    }
    Ok(ctx)
}

fn finish_eager_extension_outputs(
    ctx: Arc<EagerRuntime>,
    op: StdTensorOp,
    inputs: &[&EagerTensor],
    outputs: Vec<Tensor>,
    session: Option<&mut EagerSession<'_>>,
) -> Result<Vec<EagerTensor>> {
    if outputs.len() != op.output_count() {
        return Err(Error::Internal(format!(
            "expected {} eager outputs for {:?}, got {}",
            op.output_count(),
            op,
            outputs.len()
        )));
    }

    if !eager_grad_recording_enabled()
        || (!eager_capture_active() && !inputs.iter().any(|input| input.requires_grad))
    {
        return outputs
            .into_iter()
            .map(|output| EagerTensor::new_untracked_result(Arc::clone(&ctx), output))
            .collect();
    }

    let output_refs: Vec<&Tensor> = outputs.iter().collect();
    let recorded = match session {
        Some(session) => session.record_outputs(&op, &output_refs, inputs)?,
        None => record_eager_outputs(&op, &output_refs, inputs)?,
    };
    if recorded.traces.len() != outputs.len() {
        return Err(Error::Internal(format!(
            "expected {} eager traces for {:?}, got {}",
            outputs.len(),
            op,
            recorded.traces.len()
        )));
    }
    let results = recorded
        .traces
        .into_iter()
        .zip(recorded.semantic_traces)
        .zip(outputs)
        .map(|((trace, semantic_trace), output)| {
            if trace.requires_grad {
                EagerTensor::new_result_with_semantic_trace(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
                )
            } else {
                EagerTensor::new_unregistered_result_with_semantic_trace(
                    Arc::clone(&ctx),
                    trace.key,
                    output,
                    trace.requires_grad,
                    trace.trace,
                    semantic_trace,
                )
            }
        })
        .collect::<Result<Vec<_>>>()?;
    crate::eager::finish_residuals(&op, inputs, &results.iter().collect::<Vec<_>>())?;
    Ok(results)
}
