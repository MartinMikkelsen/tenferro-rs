use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{AllocationDomainId, AllocationId};

use super::span::{RootBoundSpan, RootResourceExtent, SpanValidationError};

/// Domain-qualified diagnostic identity. It is not an ownership proof.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationKey {
    domain: AllocationDomainId,
    local: AllocationId,
}

impl AllocationKey {
    #[doc(hidden)]
    pub const fn new(domain: AllocationDomainId, local: AllocationId) -> Self {
        Self { domain, local }
    }

    #[doc(hidden)]
    pub const fn domain(self) -> AllocationDomainId {
        self.domain
    }

    #[doc(hidden)]
    pub const fn local(self) -> AllocationId {
        self.local
    }
}

/// Private non-reused provenance for one provider root.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RootResourceId(NonZeroU64);

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RootResourceIdError {
    #[error("root-resource identity space exhausted")]
    Exhausted,
}

impl RootResourceId {
    fn fresh() -> Result<Self, RootResourceIdError> {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);

        let value = NEXT_ID
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                if current == 0 {
                    None
                } else {
                    Some(if current == u64::MAX { 0 } else { current + 1 })
                }
            })
            .map_err(|_| RootResourceIdError::Exhausted)?;
        NonZeroU64::new(value)
            .map(Self)
            .ok_or(RootResourceIdError::Exhausted)
    }

    pub(crate) const fn get(self) -> NonZeroU64 {
        self.0
    }

    /// Verify that a span was bound to this root resource.
    pub(crate) fn validate_bound_span(
        self,
        span: &RootBoundSpan,
    ) -> Result<(), SpanValidationError> {
        let actual = span.root_resource();
        if actual != self {
            return Err(SpanValidationError::DifferentRoot {
                expected: self,
                actual,
            });
        }
        Ok(())
    }
}

/// Root provenance paired with its exact checked extent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RootResourceIdentity {
    root_resource: RootResourceId,
    extent: RootResourceExtent,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum RootResourceIdentityError {
    #[error("root-resource extent is invalid: {0}")]
    Extent(#[source] SpanValidationError),
    #[error("root-resource identity space is exhausted: {0}")]
    Id(#[source] RootResourceIdError),
}

impl RootResourceIdentity {
    pub(crate) fn try_new(extent: RootResourceExtent) -> Result<Self, RootResourceIdentityError> {
        extent
            .validate()
            .map_err(RootResourceIdentityError::Extent)?;
        let root_resource = RootResourceId::fresh().map_err(RootResourceIdentityError::Id)?;
        Ok(Self {
            root_resource,
            extent,
        })
    }

    /// Rebuild the identity from a span's root-resource id and the root pin's
    /// extent.
    ///
    /// The pin owns the extent; a span only ever carries the id (#1823 G). Both
    /// values originate in one root import, so the pair is the same identity the
    /// import minted.
    pub(crate) const fn from_parts(
        root_resource: RootResourceId,
        extent: RootResourceExtent,
    ) -> Self {
        Self {
            root_resource,
            extent,
        }
    }

    pub(crate) const fn root_resource(self) -> RootResourceId {
        self.root_resource
    }

    pub(crate) const fn extent(self) -> RootResourceExtent {
        self.extent
    }

    pub(crate) fn validate_bound_span(
        self,
        span: &RootBoundSpan,
    ) -> Result<(), SpanValidationError> {
        self.root_resource.validate_bound_span(span)
    }
}
