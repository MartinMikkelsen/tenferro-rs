pub(crate) mod backend;
mod linalg;

#[cfg(feature = "cpu-faer")]
mod tlinalg;

#[cfg(feature = "cpu-blas")]
mod tlinalg_blas;

#[cfg(feature = "cpu-blas")]
mod tlinalg_workspace;

#[cfg(any(feature = "cpu-faer", feature = "cpu-blas"))]
mod tlinalg_error;

#[cfg(test)]
mod tests;
