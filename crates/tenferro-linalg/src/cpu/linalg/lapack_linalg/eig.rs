use num_complex::{Complex32, Complex64};
use tenferro_cpu::linalg_interop::BufferPool;
use tenferro_tensor::{DType, Tensor, TypedTensor};
use tlinalg_blas::Op;

use super::helpers::{
    matrix_with_batch_shape, pooled_output, provider, square_core_and_batch_result,
    tensor_from_vec_with_template, vector_with_batch_shape, zero_dim_eig_outputs, LapackLinalg,
};
use super::unsupported_dtype;

/// Eigenvalues (and with `vectors`, right eigenvectors) of a batch, as erased complex tensors.
fn eig_typed<T: LapackLinalg>(
    buffers: &mut BufferPool,
    input: &TypedTensor<T>,
    vectors: bool,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let (op, provider_op) = if vectors {
        ("eig", Op::Eig)
    } else {
        ("eig_values", Op::EigValues)
    };
    let (n, batch_shape) = square_core_and_batch_result(input, op)?;
    let values_shape = vector_with_batch_shape(n, batch_shape);
    let vectors_shape = if vectors {
        matrix_with_batch_shape(n, n, batch_shape)
    } else {
        Vec::new()
    };
    let mut values = pooled_output::<T::ComplexScalar>(buffers, op, "values", &values_shape)?;
    let mut vector_data = if vectors {
        pooled_output::<T::ComplexScalar>(buffers, op, "vectors", &vectors_shape)?
    } else {
        Vec::new()
    };
    let view = input.as_view();
    T::eig(
        buffers,
        provider_op,
        super::super::raw_view(op, &view)?,
        &mut values,
        vectors.then_some(&mut vector_data),
    )
    .map_err(provider(provider_op))?;
    let mut outputs = vec![T::wrap_complex(tensor_from_vec_with_template(
        values_shape,
        values,
        input,
    )?)];
    if vectors {
        outputs.push(T::wrap_complex(tensor_from_vec_with_template(
            vectors_shape,
            vector_data,
            input,
        )?));
    }
    Ok(outputs)
}

fn eig_erased(
    buffers: &mut BufferPool,
    input: &Tensor,
    vectors: bool,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let op = if vectors { "eig" } else { "eig_values" };
    if input.shape().contains(&0) {
        return zero_dim_eig_outputs(input, op, vectors);
    }
    let unsupported = || unsupported_dtype(op, input.dtype());
    match input.dtype() {
        DType::F32 => eig_typed(
            buffers,
            input.as_typed::<f32>().ok_or_else(unsupported)?,
            vectors,
        ),
        DType::F64 => eig_typed(
            buffers,
            input.as_typed::<f64>().ok_or_else(unsupported)?,
            vectors,
        ),
        DType::C32 => eig_typed(
            buffers,
            input.as_typed::<Complex32>().ok_or_else(unsupported)?,
            vectors,
        ),
        DType::C64 => eig_typed(
            buffers,
            input.as_typed::<Complex64>().ok_or_else(unsupported)?,
            vectors,
        ),
        _ => Err(unsupported()),
    }
}

pub(crate) fn eig(
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Vec<Tensor>> {
    eig_erased(buffers, input, true)
}

pub(crate) fn eig_values(
    buffers: &mut BufferPool,
    input: &Tensor,
) -> tenferro_tensor::Result<Tensor> {
    eig_erased(buffers, input, false)?.pop().ok_or_else(|| {
        tenferro_tensor::Error::runtime_state("eig_values", "eigenvalue output missing")
    })
}
