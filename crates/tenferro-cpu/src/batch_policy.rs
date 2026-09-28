//! Overridable batch execution strategy for batched CPU operations (#1938 D9).
//!
//! A batched operation (strided-batched or grouped GEMM, packed LU/solve) runs
//! its independent items one of several ways. [`CpuBatchStrategy::Auto`] keeps
//! the backend heuristics; every other strategy forces one route and fails with
//! a typed error when that route is not available, instead of falling back.
//!
//! Precedence is per-operation > scoped override > backend default. The backend
//! default is set with [`crate::CpuBackend::with_batch_policy`]; a scoped
//! override, which also expresses a per-operation choice when it wraps a single
//! call, is [`crate::with_batch_policy`]. Scopes nest and the
//! innermost wins; each restores the previous policy on return, error and
//! unwind. Nothing here mutates process-global state.
//!
//! # Examples
//!
//! ```rust
//! use tenferro_cpu::{CpuBackend, CpuBatchPolicy, CpuBatchStrategy};
//!
//! let backend = CpuBackend::with_threads(1)?
//!     .with_batch_policy(CpuBatchPolicy::new(CpuBatchStrategy::Sequential));
//! assert_eq!(backend.batch_policy().strategy(), CpuBatchStrategy::Sequential);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

/// How the items of one batched operation are executed.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBatchStrategy;
///
/// assert_eq!(CpuBatchStrategy::default(), CpuBatchStrategy::Auto);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum CpuBatchStrategy {
    /// Choose a supported route from the thresholds and the provider.
    #[default]
    Auto,
    /// Sequential batch, sequential items: no parallelism at all.
    Sequential,
    /// Outer-parallel batch: tenferro lanes, each running items sequentially.
    OuterParallel,
    /// Sequential batch whose items may use the provider's own parallelism.
    ProviderItems,
    /// One vendor batch call for the whole batch, with no tenferro fan-out.
    WholeBatchVendor,
}

/// Thresholds that [`CpuBatchStrategy::Auto`] applies.
///
/// The defaults are the values the backend used before these knobs existed;
/// they are not tuned results.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::CpuBatchThresholds;
///
/// let thresholds = CpuBatchThresholds::default()
///     .with_vendor_batch_max_item_dim(8)
///     .with_outer_min_items(4);
/// assert_eq!(thresholds.vendor_batch_max_item_dim(), 8);
/// assert_eq!(thresholds.outer_min_items(), 4);
/// assert_eq!(thresholds.outer_min_items_per_lane(), 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CpuBatchThresholds {
    vendor_batch_max_item_dim: usize,
    outer_min_items: usize,
    outer_min_items_per_lane: usize,
}

impl Default for CpuBatchThresholds {
    fn default() -> Self {
        Self {
            // Per-item work: the BLAS provider's measured small-job cutoff.
            vendor_batch_max_item_dim: 16,
            // Total batch work: fan-out needs more than one item.
            outer_min_items: 2,
            // Chunk granularity: every lane must receive at least one item.
            outer_min_items_per_lane: 1,
        }
    }
}

impl CpuBatchThresholds {
    /// Per-item work: the largest `m`, `n` and `k` for which `Auto` lets a
    /// vendor batch call (`cblas_?gemm_batch`) handle a grouped GEMM batch.
    ///
    /// Strided batched contractions do not use this cutoff: under `Auto` they
    /// keep one provider GEMM per item, and reach the vendor batch call only
    /// through [`CpuBatchStrategy::WholeBatchVendor`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(tenferro_cpu::CpuBatchThresholds::default().vendor_batch_max_item_dim(), 16);
    /// ```
    #[must_use]
    pub fn vendor_batch_max_item_dim(&self) -> usize {
        self.vendor_batch_max_item_dim
    }

    /// Total batch work: the fewest items for which `Auto` fans out.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(tenferro_cpu::CpuBatchThresholds::default().outer_min_items(), 2);
    /// ```
    #[must_use]
    pub fn outer_min_items(&self) -> usize {
        self.outer_min_items
    }

    /// Chunk granularity: the fewest items each lane must receive before
    /// `Auto` fans a batch out over every lane.
    ///
    /// # Examples
    ///
    /// ```rust
    /// assert_eq!(tenferro_cpu::CpuBatchThresholds::default().outer_min_items_per_lane(), 1);
    /// ```
    #[must_use]
    pub fn outer_min_items_per_lane(&self) -> usize {
        self.outer_min_items_per_lane
    }

    /// Return these thresholds with a different vendor-batch item limit.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBatchThresholds;
    /// let thresholds = CpuBatchThresholds::default().with_vendor_batch_max_item_dim(0);
    /// assert_eq!(thresholds.vendor_batch_max_item_dim(), 0);
    /// ```
    #[must_use]
    pub fn with_vendor_batch_max_item_dim(mut self, limit: usize) -> Self {
        self.vendor_batch_max_item_dim = limit;
        self
    }

    /// Return these thresholds with a different minimum fan-out batch size.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBatchThresholds;
    /// let thresholds = CpuBatchThresholds::default().with_outer_min_items(64);
    /// assert_eq!(thresholds.outer_min_items(), 64);
    /// ```
    #[must_use]
    pub fn with_outer_min_items(mut self, items: usize) -> Self {
        self.outer_min_items = items;
        self
    }

    /// Return these thresholds with a different minimum chunk per lane.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBatchThresholds;
    /// let thresholds = CpuBatchThresholds::default().with_outer_min_items_per_lane(8);
    /// assert_eq!(thresholds.outer_min_items_per_lane(), 8);
    /// ```
    #[must_use]
    pub fn with_outer_min_items_per_lane(mut self, items: usize) -> Self {
        self.outer_min_items_per_lane = items;
        self
    }

    /// Whether `Auto` fans `items` out over `lanes` lanes: more than one lane,
    /// at least [`Self::outer_min_items`] items, and at least
    /// [`Self::outer_min_items_per_lane`] items for every lane.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::CpuBatchThresholds;
    ///
    /// let thresholds = CpuBatchThresholds::default();
    /// assert!(thresholds.fans_out(8, 4));
    /// assert!(!thresholds.fans_out(3, 4));
    /// assert!(!thresholds.fans_out(8, 1));
    /// ```
    #[must_use]
    pub fn fans_out(&self, items: usize, lanes: usize) -> bool {
        lanes > 1
            && items >= self.outer_min_items
            && items >= lanes.saturating_mul(self.outer_min_items_per_lane)
    }

    /// Whether `Auto` may hand every GEMM of dimensions `dims` to one vendor
    /// batch call.
    pub(crate) fn auto_uses_vendor_batch(
        &self,
        dims: impl IntoIterator<Item = [usize; 3]>,
    ) -> bool {
        let limit = self.vendor_batch_max_item_dim;
        let mut count = 0usize;
        for [m, n, k] in dims {
            if m > limit || n > limit || k > limit {
                return false;
            }
            count += 1;
        }
        count > 1
    }
}

/// A batch strategy together with the thresholds `Auto` uses.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy, CpuBatchThresholds};
///
/// let policy = CpuBatchPolicy::default()
///     .with_thresholds(CpuBatchThresholds::default().with_outer_min_items(8));
/// assert_eq!(policy.strategy(), CpuBatchStrategy::Auto);
/// assert_eq!(policy.thresholds().outer_min_items(), 8);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CpuBatchPolicy {
    strategy: CpuBatchStrategy,
    thresholds: CpuBatchThresholds,
}

impl CpuBatchPolicy {
    /// A policy with `strategy` and the default thresholds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};
    /// let policy = CpuBatchPolicy::new(CpuBatchStrategy::WholeBatchVendor);
    /// assert_eq!(policy.strategy(), CpuBatchStrategy::WholeBatchVendor);
    /// ```
    #[must_use]
    pub fn new(strategy: CpuBatchStrategy) -> Self {
        Self {
            strategy,
            thresholds: CpuBatchThresholds::default(),
        }
    }

    /// Return this policy with different thresholds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuBatchPolicy, CpuBatchThresholds};
    /// let policy = CpuBatchPolicy::default().with_thresholds(CpuBatchThresholds::default());
    /// assert_eq!(policy, CpuBatchPolicy::default());
    /// ```
    #[must_use]
    pub fn with_thresholds(mut self, thresholds: CpuBatchThresholds) -> Self {
        self.thresholds = thresholds;
        self
    }

    /// The selected strategy.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuBatchPolicy, CpuBatchStrategy};
    /// assert_eq!(CpuBatchPolicy::default().strategy(), CpuBatchStrategy::Auto);
    /// ```
    #[must_use]
    pub fn strategy(&self) -> CpuBatchStrategy {
        self.strategy
    }

    /// The thresholds `Auto` applies.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_cpu::{CpuBatchPolicy, CpuBatchThresholds};
    /// assert_eq!(CpuBatchPolicy::default().thresholds(), CpuBatchThresholds::default());
    /// ```
    #[must_use]
    pub fn thresholds(&self) -> CpuBatchThresholds {
        self.thresholds
    }
}

/// Run `f` on `session` with `policy` as the effective batch policy.
///
/// This is the scoped override of the precedence per-operation > scoped >
/// backend default; wrapping a single call expresses a per-operation choice.
/// The previous policy is restored when `f` returns, returns an error or
/// unwinds, and scopes nest with the innermost winning.
///
/// # Examples
///
/// ```rust
/// use tenferro_cpu::{with_batch_policy, CpuBackend, CpuBatchPolicy, CpuBatchStrategy};
/// use tenferro_tensor::{BackendSessionHost, DotGeneralConfig, Tensor, TensorRead};
///
/// let mut backend = CpuBackend::with_threads(1)?;
/// let lhs = Tensor::from_vec_col_major(vec![2, 2, 3], vec![1.0_f64; 12])?;
/// let rhs = Tensor::from_vec_col_major(vec![2, 2, 3], vec![2.0_f64; 12])?;
/// let config = DotGeneralConfig {
///     lhs_contracting_dims: [1].as_slice().into(),
///     rhs_contracting_dims: [0].as_slice().into(),
///     lhs_batch_dims: [2].as_slice().into(),
///     rhs_batch_dims: [2].as_slice().into(),
/// };
/// let product = backend.with_backend_session(|session| {
///     // ProviderItems runs one provider GEMM per item with every provider;
///     // a forced Sequential is rejected by providers with their own threading.
///     with_batch_policy(session, CpuBatchPolicy::new(CpuBatchStrategy::ProviderItems), |session| {
///         session.dot_general_read(
///             TensorRead::from_tensor(&lhs),
///             TensorRead::from_tensor(&rhs),
///             &config,
///         )
///     })
/// })???;
/// assert_eq!(product.as_slice::<f64>()?, &[4.0; 12]);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// # Errors
///
/// Returns [`tenferro_tensor::Error::Unsupported`] without running `f` when
/// `session` is neither a CPU execution session nor a session that forwards
/// one through [`tenferro_tensor::BackendSession::native_session`] (a CUDA or
/// other custom session). `f`'s own result is returned unchanged inside `Ok`.
///
/// # Panics
///
/// A panic in `f` propagates after the previous policy is restored.
pub fn with_batch_policy<R>(
    session: &mut dyn tenferro_tensor::BackendSession,
    policy: CpuBatchPolicy,
    f: impl FnOnce(&mut dyn tenferro_tensor::BackendSession) -> R,
) -> tenferro_tensor::Result<R> {
    let Some(previous) =
        crate::with_cpu_exec_session(session, |cpu| cpu.replace_batch_policy(policy))
    else {
        return Err(tenferro_tensor::Error::unsupported(
            "tenferro_cpu::with_batch_policy",
            "the session is not a CPU execution session and does not forward one; CPU batch \
             policies apply only to CpuBackend sessions",
        ));
    };
    // `f` runs on the caller's own session, so a wrapping session keeps its
    // overrides inside the scope. The policy is restored before an unwind
    // continues; nothing observes the session between the panic and the
    // restore, so asserting unwind safety is sound.
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&mut *session)));
    // INVARIANT: the same session yielded a CPU execution session above and
    // native tokens are stable for a session's lifetime, so this visit runs.
    let _ = crate::with_cpu_exec_session(session, |cpu| cpu.replace_batch_policy(previous));
    match outcome {
        Ok(value) => Ok(value),
        Err(payload) => std::panic::resume_unwind(payload),
    }
}

/// Typed error for a forced batch strategy with no route for this operation.
pub(crate) fn strategy_unavailable(
    op: &'static str,
    strategy: CpuBatchStrategy,
    reason: &str,
) -> tenferro_tensor::Error {
    tenferro_tensor::Error::unsupported(
        op,
        format!(
            "batch strategy {strategy:?} is not available: {reason}; use CpuBatchStrategy::Auto or another strategy"
        ),
    )
}
