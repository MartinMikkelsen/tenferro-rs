use tenferro_tensor::{TypedTensor, TypedTensorView};

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

/// The storage offset of batch item `index` of a rank-`2 + B` view (first batch axis fastest).
fn item_offset<T: 'static>(view: &TypedTensorView<'_, T>, index: usize) -> isize {
    let mut rest = index;
    let mut offset = view.offset();
    for (&dim, &stride) in view.shape()[2..].iter().zip(&view.strides()[2..]) {
        // INVARIANT: `index < batch` and every batch dim is nonzero here, so `rest % dim` is a
        // valid index whose offset the view's own construction validated.
        offset += (rest % dim) as isize * stride;
        rest /= dim;
    }
    offset
}

/// Reject a batch with a non-finite entry before any item is factored.
///
/// Both CPU routes run this screen first, so a non-finite input is reported identically and ahead
/// of any provider failure, whichever route serves the call.
pub(crate) fn screen_non_finite<T: Copy + 'static>(
    op: &'static str,
    view: &TypedTensorView<'_, T>,
    batch: usize,
    is_finite: impl Fn(T) -> bool,
) -> tenferro_tensor::Result<()> {
    let storage = view.host_storage()?;
    let (m, n) = (view.shape()[0], view.shape()[1]);
    let (row_stride, col_stride) = (view.strides()[0], view.strides()[1]);
    for index in 0..batch {
        let base = item_offset(view, index);
        for col in 0..n {
            for row in 0..m {
                // INVARIANT: the view validated every reachable offset against its storage.
                let value = storage
                    [(base + row as isize * row_stride + col as isize * col_stride) as usize];
                if !is_finite(value) {
                    return Err(crate::error::into_tensor_error(
                        op,
                        crate::Error::NonFinite { op, role: "input" },
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The rank of every item of a batched `R` (`k x n` per item), by [`prefix_rank`].
///
/// An all-zero item has a zero diagonal and so rank zero; an empty `R` (`k == 0` or `n == 0`) is
/// rank zero too.
pub(crate) fn batch_ranks<T: Copy>(
    r: &[T],
    k: usize,
    n: usize,
    batch: usize,
    options: RankRevealingQrOptions,
    magnitude: impl Fn(T) -> f64,
) -> tenferro_tensor::Result<Vec<i64>> {
    let mut ranks = Vec::with_capacity(batch);
    if k > 0 && n > 0 {
        for r_item in r.chunks_exact(k * n) {
            ranks.push(prefix_rank(
                (0..k).map(|diagonal| magnitude(r_item[diagonal + diagonal * k])),
                options,
            )?);
        }
    }
    ranks.resize(batch, 0);
    Ok(ranks)
}
