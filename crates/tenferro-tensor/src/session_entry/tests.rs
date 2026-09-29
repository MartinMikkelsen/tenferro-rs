use std::error::Error as _;

use super::SessionEntryError;
use crate::ErrorKind;

#[test]
fn every_entry_failure_names_its_backend_and_is_runtime_state() {
    let executor = std::io::Error::other("rayon pool unavailable");
    let errors = [
        SessionEntryError::Reentered { backend: "A" },
        SessionEntryError::Contended {
            backend: "B",
            message: "caller-managed domain busy".into(),
        },
        SessionEntryError::IncompatibleContext {
            backend: "C",
            message: "scope witness mismatch".into(),
        },
        SessionEntryError::ResourcePoisoned {
            backend: "D",
            resource: "the CPU resource arbiter",
        },
        SessionEntryError::Executor {
            backend: "E",
            source: Box::new(executor),
        },
    ];
    let backends: Vec<_> = errors.iter().map(SessionEntryError::backend).collect();
    assert_eq!(backends, ["A", "B", "C", "D", "E"]);
    for error in &errors {
        assert_eq!(error.kind(), ErrorKind::RuntimeState);
        assert!(error.to_string().starts_with(error.backend()));
    }
    // Only the executor failure carries a typed source.
    assert!(errors[4].source().is_some());
    assert!(errors[..4].iter().all(|error| error.source().is_none()));
}
