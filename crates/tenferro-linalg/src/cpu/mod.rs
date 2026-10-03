pub(crate) mod backend;
mod linalg;

#[cfg(feature = "cpu-faer")]
mod tlinalg;

#[cfg(test)]
mod tests;
