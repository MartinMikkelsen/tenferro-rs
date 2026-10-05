use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::TypedTensor;
use tlinalg_blas::Op;

use super::helpers::{
    batch_element_count, has_zero_dim, matrix_core_and_batch_result, matrix_with_batch_shape,
    pooled_output, provider, square_core_and_batch_result, tensor_from_vec_with_template,
    LapackLinalg,
};

/// Explicit complete-pivot LU, `P A Qᵀ = L U`, of every square matrix of a batch.
pub(crate) fn full_piv_lu<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "full_piv_lu";
    let (n, batch_shape) = square_core_and_batch_result(input, OP)?;
    let batch = batch_element_count(OP, batch_shape)?;
    let shape = matrix_with_batch_shape(n, n, batch_shape);
    if has_zero_dim(input.shape()) {
        let empty = || tensor_from_vec_with_template(shape.clone(), Vec::new(), input);
        return Ok(vec![
            empty()?,
            empty()?,
            empty()?,
            empty()?,
            tensor_from_vec_with_template(batch_shape.to_vec(), vec![T::unit(); batch], input)?,
        ]);
    }
    let mut p = pooled_output::<T>(buffers, OP, "permutation matrix", &shape)?;
    let mut l = pooled_output::<T>(buffers, OP, "lower factor", &shape)?;
    let mut u = pooled_output::<T>(buffers, OP, "upper factor", &shape)?;
    let mut q = pooled_output::<T>(buffers, OP, "permutation matrix", &shape)?;
    let mut parity = buffers.acquire_with_capacity::<T>(batch);
    let view = input.as_view();
    T::full_piv_lu(
        buffers,
        super::super::raw_view(OP, &view)?,
        tlinalg_blas::full_piv_lu::FullPivLuOutputs {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            q: &mut q,
            parity: &mut parity,
        },
    )
    .map_err(provider(Op::FullPivLu))?;
    Ok(vec![
        tensor_from_vec_with_template(shape.clone(), p, input)?,
        tensor_from_vec_with_template(shape.clone(), l, input)?,
        tensor_from_vec_with_template(shape.clone(), u, input)?,
        tensor_from_vec_with_template(shape, q, input)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity, input)?,
    ])
}

pub(crate) fn full_piv_lu_solve<T: LapackLinalg>(
    buffers: &mut BufferPool,
    a: &TypedTensor<T>,
    b: &TypedTensor<T>,
    transpose_a: bool,
) -> tenferro_tensor::Result<TypedTensor<T>> {
    const OP: &str = "full_piv_lu_solve";
    let (n, a_batch_shape) = square_core_and_batch_result(a, OP)?;
    let (b_rows, _, b_batch_shape) = matrix_core_and_batch_result(b, OP)?;
    if b_rows != n {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            vec![n],
            vec![b_rows],
        ));
    }
    if a_batch_shape != b_batch_shape {
        return Err(tenferro_tensor::Error::shape_mismatch(
            OP,
            a_batch_shape.to_vec(),
            b_batch_shape.to_vec(),
        ));
    }
    if has_zero_dim(a.shape()) || has_zero_dim(b.shape()) {
        return tensor_from_vec_with_template(b.shape().to_vec(), Vec::new(), b);
    }
    let mut output = pooled_output::<T>(buffers, OP, "rhs", b.shape())?;
    let (a_view, b_view) = (a.as_view(), b.as_view());
    T::full_piv_lu_solve(
        buffers,
        transpose_a,
        super::super::raw_view(OP, &a_view)?,
        super::super::raw_view(OP, &b_view)?,
        &mut output,
    )
    .map_err(provider(Op::FullPivLuSolve))?;
    tensor_from_vec_with_template(b.shape().to_vec(), output, b)
}
