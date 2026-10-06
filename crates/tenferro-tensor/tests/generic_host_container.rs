use tenferro_tensor::{Rank, TypedTensor, TypedTensorViewMut};

#[test]
fn non_copy_host_values_support_explicit_import_copy_and_checked_access() {
    let values = ["a", "b", "c", "d", "e", "f"].map(str::to_owned).to_vec();
    let mut owner = TypedTensor::<String, Rank<2>>::from_vec_row_major([2, 3], values).unwrap();
    assert_eq!(owner.get(&[1, 0]).unwrap(), "d");
    assert_eq!(owner.get(&[0, 2]).unwrap(), "c");
    assert!(owner.get(&[2, 0]).is_err());
    let mut copy = owner.duplicate().unwrap();
    assert_ne!(
        owner.host_data().unwrap().as_ptr(),
        copy.host_data().unwrap().as_ptr()
    );
    *copy.get_mut(&[0, 2]).unwrap() = "changed".to_owned();
    assert_eq!(owner.get(&[0, 2]).unwrap(), "c");
    assert_eq!(copy.get(&[0, 2]).unwrap(), "changed");
    *owner.get_mut(&[1, 2]).unwrap() = "last".to_owned();
    let transposed = owner.as_view().transpose_view([1, 0]).unwrap();
    assert_eq!(
        transposed.to_col_major().unwrap().host_data().unwrap(),
        &["a", "b", "c", "d", "e", "last"]
    );
    let mut storage = ["x".to_owned(), "y".to_owned(), "z".to_owned()];
    let view =
        TypedTensorViewMut::<_, Rank<1>>::from_slice_ranked([2], [-1], 2, &mut storage).unwrap();
    assert_eq!(
        view.to_col_major().unwrap().host_data().unwrap(),
        &["z", "y"]
    );
}

#[test]
fn non_copy_host_import_handles_scalar_empty_and_rejected_shape() {
    let scalar = TypedTensor::<String>::from_vec_row_major([], vec!["scalar".to_owned()]).unwrap();
    assert_eq!(scalar.get(&[]).unwrap(), "scalar");
    let empty = TypedTensor::<String, Rank<2>>::from_vec_row_major([0, 3], vec![]).unwrap();
    assert!(empty.host_data().unwrap().is_empty());
    assert!(TypedTensor::<String>::from_vec_row_major([2, 3], vec![String::new(); 5]).is_err());
    assert!(TypedTensor::<String>::from_vec_row_major([usize::MAX, 2], vec![]).is_err());
}

/// #1984: the dtype-erased tensor imports row-major data explicitly.
#[test]
fn erased_tensor_imports_row_major_data_in_column_major_storage() {
    use tenferro_tensor::{ErrorKind, Tensor};

    // Row-major [2, 3, 2]: value = 6 i + 2 j + k.
    let data: Vec<f64> = (0..12).map(f64::from).collect();
    let tensor = Tensor::from_vec_row_major(vec![2, 3, 2], data).unwrap();
    let col_major = tensor.as_slice::<f64>().unwrap();
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..2 {
                assert_eq!(col_major[i + 2 * j + 6 * k], (6 * i + 2 * j + k) as f64);
            }
        }
    }

    let scalar = Tensor::from_vec_row_major(Vec::<usize>::new(), vec![7_i32]).unwrap();
    assert_eq!(scalar.as_slice::<i32>().unwrap(), &[7]);
    let empty = Tensor::from_vec_row_major(vec![0, 3], Vec::<f32>::new()).unwrap();
    assert_eq!(empty.shape(), &[0, 3]);
    let error = Tensor::from_vec_row_major(vec![2, 3], vec![1.0_f64; 5]).unwrap_err();
    assert!(matches!(error.kind(), ErrorKind::Validation(_)));
}
