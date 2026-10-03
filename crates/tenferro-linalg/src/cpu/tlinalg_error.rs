//! Shared translation of a `tlinalg` failure into tenferro's error vocabulary.
//!
//! Both provider routes report [`tlinalg_traits::Error`], so the mapping lives here rather than in
//! either adapter. The mapping is one-to-one on kind, role and typed source, because callers
//! downcast the source and classify by kind.

#![cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]

use tlinalg_traits::Error as TlError;
use tlinalg_traits::Op;

/// Rebuild tenferro's error from a `tlinalg` failure.
///
/// The mapping is one-to-one on kind, role and typed source, because callers downcast the source
/// and classify by kind.
pub(crate) fn map_error(op: Op, error: TlError) -> tenferro_tensor::Error {
    let op_str = op.as_str();
    match error {
        TlError::NonConvergence { .. } => {
            crate::error::into_tensor_error(op_str, crate::Error::NonConvergence { op: op_str })
        }
        TlError::NonFinite { role, .. } => crate::error::into_tensor_error(
            op_str,
            crate::Error::NonFinite {
                op: op_str,
                role: role.as_str(),
            },
        ),
        TlError::Singular { .. } => {
            crate::error::into_tensor_error(op_str, crate::Error::Singular { op: op_str })
        }
        TlError::InvalidArgument { role, detail, .. } => {
            tenferro_tensor::Error::invalid_argument(op_str, role, detail)
        }
        TlError::InvalidWorkspace {
            library,
            routine,
            detail,
            ..
        } => crate::error::invalid_workspace(op_str, library, routine, detail),
        TlError::Inconsistent { detail, .. } => {
            tenferro_tensor::Error::Internal(format!("{op_str}: {detail}"))
        }
        // `tlinalg_traits::Error` is non-exhaustive so a new variant is not a breaking change for
        // the crate that reports it. The host cannot reproduce a payload it does not know, so it
        // fails loudly instead of guessing a kind and silently misclassifying the failure.
        other => {
            tenferro_tensor::Error::Internal(format!("{op_str}: unmapped tlinalg error {other:?}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;
    use tenferro_tensor::{Error as TensorError, ErrorKind};

    fn payload(error: &TensorError) -> Option<&crate::Error> {
        error.source().and_then(|source| source.downcast_ref())
    }

    #[test]
    fn every_variant_rebuilds_tenferros_kind_role_and_source() {
        // The linalg-owned kinds carry the same downcastable payload a caller classifies on.
        let error = map_error(Op::Svd, TlError::NonConvergence { op: Op::Svd });
        assert_eq!(error.kind(), ErrorKind::NumericalFailure);
        assert!(matches!(
            payload(&error),
            Some(crate::Error::NonConvergence { op: "svd" })
        ));

        let error = map_error(
            Op::Eigh,
            TlError::NonFinite {
                op: Op::Eigh,
                role: tlinalg_traits::NonFiniteRole::RDiagonal,
            },
        );
        assert_eq!(error.kind(), ErrorKind::NumericalFailure);
        assert!(matches!(
            payload(&error),
            Some(crate::Error::NonFinite {
                op: "eigh",
                role: "R diagonal"
            })
        ));

        let error = map_error(Op::Solve, TlError::Singular { op: Op::Solve });
        assert_eq!(error.kind(), ErrorKind::NumericalFailure);
        assert!(matches!(
            payload(&error),
            Some(crate::Error::Singular { op: "solve" })
        ));

        // The provider shapes keep their role and their text, because callers match on both.
        let error = map_error(
            Op::LuFactor,
            TlError::InvalidArgument {
                op: Op::LuFactor,
                role: "lapack_argument",
                detail: "LAPACK getrf argument 3 had an illegal value".to_owned(),
            },
        );
        let rendered = error.to_string();
        assert!(rendered.contains("lapack_argument"), "{rendered}");
        assert!(rendered.contains("getrf argument 3"), "{rendered}");

        let error = map_error(
            Op::Svd,
            TlError::InvalidWorkspace {
                op: Op::Svd,
                library: "LAPACK",
                routine: "dgesdd",
                detail: "query was zero".to_owned(),
            },
        );
        assert_eq!(error.kind(), ErrorKind::BackendFailure);
        let rendered = error.to_string();
        assert!(rendered.contains("dgesdd"), "{rendered}");
        assert!(rendered.contains("query was zero"), "{rendered}");

        let error = map_error(
            Op::LuSolvePrepared,
            TlError::Inconsistent {
                op: Op::LuSolvePrepared,
                detail: "different batches",
            },
        );
        assert!(matches!(error, TensorError::Internal(_)));
        assert!(error.to_string().contains("different batches"));
    }
}
