use tenferro_tensor::TypedTensor;

use crate::{RankRevealingQrOptions, RankRevealingQrResult};

pub(crate) type TypedRrqr<T> = RankRevealingQrResult<TypedTensor<T>, TypedTensor<i64>>;

pub(crate) fn prefix_rank(
    diagonal_magnitudes: impl IntoIterator<Item = f64>,
    options: RankRevealingQrOptions,
) -> tenferro_tensor::Result<i64> {
    let diagonal = diagonal_magnitudes.into_iter().collect::<Vec<_>>();
    if diagonal.iter().any(|value| !value.is_finite()) {
        return Err(crate::error::into_tensor_error(
            "rank_revealing_qr",
            crate::Error::NonFinite {
                op: "rank_revealing_qr",
                role: "R diagonal",
            },
        ));
    }
    let Some(&leading) = diagonal.first() else {
        return Ok(0);
    };
    // A finite rtol times a finite leading magnitude can overflow to infinity;
    // that means no diagonal clears the requested threshold, hence rank zero.
    let threshold = options.atol.max(options.rtol * leading);
    let rank = diagonal
        .iter()
        .take_while(|&&value| value > threshold)
        .count();
    i64::try_from(rank).map_err(|_| {
        tenferro_tensor::Error::invalid_argument(
            "rank_revealing_qr",
            "rank",
            "rank exceeds i64 range",
        )
    })
}

/// The identity column permutation of `n` columns, for an item that is not factored.
#[cfg(feature = "cpu-faer")]
pub(crate) fn identity_permutation(n: usize) -> tenferro_tensor::Result<Vec<i64>> {
    (0..n)
        .map(|column| {
            i64::try_from(column).map_err(|_| {
                tenferro_tensor::Error::invalid_argument(
                    "rank_revealing_qr",
                    "column_permutation",
                    "column index exceeds i64 range",
                )
            })
        })
        .collect()
}
