use tenferro_cpu::linalg_interop::{BufferPool, PoolScalar};
use tenferro_tensor::TypedTensor;
use tlinalg_blas::Op;

use super::helpers::{
    batch_element_count, checked_product, has_zero_dim, matrix_core_and_batch_result,
    matrix_with_batch_shape, pooled_output, provider, tensor_from_vec_with_template,
    vector_with_batch_shape, LapackLinalg,
};

/// Explicit partial-pivot LU, `P A = L U`, of every matrix of a batch.
pub(crate) fn lu<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<Vec<TypedTensor<T>>> {
    const OP: &str = "lu";
    let (m, n, batch_shape) = matrix_core_and_batch_result(input, OP)?;
    let k = m.min(n);
    let batch = batch_element_count(OP, batch_shape)?;
    let p_shape = matrix_with_batch_shape(m, m, batch_shape);
    let l_shape = matrix_with_batch_shape(m, k, batch_shape);
    let u_shape = matrix_with_batch_shape(k, n, batch_shape);
    if has_zero_dim(input.shape()) {
        return Ok(vec![
            tensor_from_vec_with_template(p_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(l_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(u_shape, Vec::new(), input)?,
            tensor_from_vec_with_template(batch_shape.to_vec(), vec![T::unit(); batch], input)?,
        ]);
    }
    let mut p = pooled_output::<T>(buffers, OP, "permutation matrix", &p_shape)?;
    let mut l = pooled_output::<T>(buffers, OP, "lower factor", &l_shape)?;
    let mut u = pooled_output::<T>(buffers, OP, "upper factor", &u_shape)?;
    let mut parity = buffers.acquire_with_capacity::<T>(batch);
    let view = input.as_view();
    T::lu(
        buffers,
        super::super::raw_view(OP, &view)?,
        tlinalg_blas::lu::LuOutputs {
            p: &mut p,
            l: &mut l,
            u: &mut u,
            parity: &mut parity,
        },
    )
    .map_err(provider(Op::Lu))?;
    Ok(vec![
        tensor_from_vec_with_template(p_shape, p, input)?,
        tensor_from_vec_with_template(l_shape, l, input)?,
        tensor_from_vec_with_template(u_shape, u, input)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity, input)?,
    ])
}

pub(crate) fn lu_factor<T: LapackLinalg>(
    ctx: &tenferro_cpu::CpuExecutionContext<'_>,
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
) -> tenferro_tensor::Result<(TypedTensor<T>, TypedTensor<i32>, TypedTensor<T>)> {
    let (m, n, batch_shape) = matrix_core_and_batch_result(input, "lu_factor")?;
    let k = m.min(n);
    let batch_total = batch_element_count("lu_factor", batch_shape)?;
    if has_zero_dim(input.shape()) {
        return Ok((
            tensor_from_vec_with_template(input.shape().to_vec(), Vec::new(), input)?,
            tensor_from_vec_with_template(
                vector_with_batch_shape(k, batch_shape),
                Vec::new(),
                input,
            )?,
            tensor_from_vec_with_template(
                batch_shape.to_vec(),
                vec![T::unit(); batch_total],
                input,
            )?,
        ));
    }

    let pivot_len = checked_product("lu_factor", "pivot output", &[k, batch_total])?;
    let mut lu_data = buffers.acquire_with_capacity::<T>(input.n_elements());
    lu_data.extend_from_slice(input.host_data()?);
    let mut pivot_data = <i32 as PoolScalar>::pool_acquire_zeroed(buffers, pivot_len);
    let mut parity_data = buffers.acquire_with_capacity::<T>(batch_total);
    parity_data.resize(batch_total, T::unit());
    crate::cpu::tlinalg_blas::factor_batch::<T>(
        ctx,
        Op::LuFactor,
        m,
        n,
        &mut lu_data,
        &mut pivot_data,
        &mut parity_data,
    )?;

    Ok((
        tensor_from_vec_with_template(input.shape().to_vec(), lu_data, input)?,
        tensor_from_vec_with_template(vector_with_batch_shape(k, batch_shape), pivot_data, input)?,
        tensor_from_vec_with_template(batch_shape.to_vec(), parity_data, input)?,
    ))
}
