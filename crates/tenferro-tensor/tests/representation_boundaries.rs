//! D1/D2/D5 representation boundaries: the statically host-owned representation,
//! group promotion, and checked representation narrowing.

use tenferro_tensor::{
    BackendStorageHandle, DType, DynRank, Dynamic, Gpu, Host, Placement, Rank, StorageBuffer,
    Tensor, TypedTensor,
};

#[test]
fn host_representation_owns_plain_elements_and_clones_independently() {
    let mut host = TypedTensor::<String, Rank<2>, Host>::from_host_vec_col_major(
        [2, 2],
        vec!["a".into(), "b".into(), "c".into(), "d".into()],
    )
    .unwrap();
    // Column-major: [0,0]=a [1,0]=b [0,1]=c [1,1]=d
    assert_eq!(host[&[1, 0]], "b");
    assert_eq!(host.get(&[0, 1]).unwrap(), "c");
    host[&[1, 0]] = "changed".into();
    assert_eq!(host.get(&[1, 0]).unwrap(), "changed");

    let copy = host.clone();
    assert_eq!(copy[&[1, 0]], "changed");
    assert_ne!(copy.as_slice().as_ptr(), host.as_slice().as_ptr());
    assert_eq!(host.shape(), &[2, 2]);
    assert_eq!(host.rank(), 2);
    assert_eq!(host.layout().strides(), &[1, 2]);
    assert_eq!(host.n_elements(), 4);
    assert!(host.is_col_major_contiguous().unwrap());
}

#[test]
fn host_representation_keeps_checked_and_panicking_access_separate() {
    let host =
        TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2]).unwrap();
    assert!(host.get(&[2]).is_err());
    assert!(host.get(&[0, 0]).is_err());
    assert_eq!(host.linear_offset(&[1]).unwrap(), 1);
    assert_eq!(host.into_layout().strides(), &[1]);
}

#[test]
#[should_panic(expected = "is invalid")]
fn host_index_panics_like_slice_indexing() {
    let host =
        TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2]).unwrap();
    let _ = host[&[5]];
}

#[test]
fn host_representation_maps_host_elements_through_owning_guards() {
    let mut host =
        TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1, 2]).unwrap();
    {
        let read = host.map_read();
        assert_eq!(&read[..], &[1, 2]);
    }
    host.map_write().copy_from_slice(&[3, 4]).unwrap();
    assert_eq!(host.as_slice(), &[3, 4]);
}

#[test]
fn row_major_import_is_explicit_and_column_major_afterwards() {
    let host =
        TypedTensor::<i32, Rank<2>, Host>::from_host_vec_row_major([2, 3], vec![1, 2, 3, 4, 5, 6])
            .unwrap();
    assert_eq!(host[&[1, 0]], 4);
    assert_eq!(host[&[0, 2]], 3);
    assert_eq!(host.as_slice(), &[1, 4, 2, 5, 3, 6]);
    assert!(
        TypedTensor::<i32, Rank<2>, Host>::from_host_vec_row_major([2, 3], vec![1, 2, 3]).is_err()
    );
}

#[test]
fn host_owner_moves_into_the_union_without_copying() {
    let host =
        TypedTensor::<i32, DynRank, Host>::from_host_vec_col_major(vec![1], vec![7]).unwrap();
    let pointer = host.as_slice().as_ptr();
    let dynamic: TypedTensor<i32, DynRank, Dynamic> = host.into_dynamic();
    assert_eq!(dynamic.host_data().unwrap().as_ptr(), pointer);
    assert_eq!(dynamic.host_data().unwrap(), &[7]);
    // The union is host-resident, so checked narrowing back succeeds.
    let narrowed = dynamic.into_host().unwrap();
    assert_eq!(narrowed.as_slice().as_ptr(), pointer);
    assert_eq!(narrowed.into_host_vec(), vec![7]);
}

#[test]
fn checked_narrowing_retains_the_source_owner_on_failure() {
    let handle = BackendStorageHandle::<f64>::new_with_len(11, 2);
    let device = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(handle)),
        Placement::default(),
    )
    .unwrap();
    assert!(device.backend_buffer().is_some());

    let Err(failure) = device.into_host() else {
        panic!("a backend buffer is not plain host storage");
    };
    assert!(!failure.error().to_string().is_empty());
    let owner = failure.into_owner();
    assert_eq!(owner.shape(), &[2]);
    assert!(owner.backend_buffer().is_some());
    // The retained owner still narrows to the group-backed representation.
    let gpu = owner.into_gpu().unwrap();
    assert!(gpu.is_backend_buffer());

    let plain = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![1.0, 2.0]).unwrap();
    let Err(failure) = plain.into_gpu() else {
        panic!("a plain host owner is not group-backed");
    };
    let retained = failure.into_owner();
    assert_eq!(retained.host_data().unwrap(), &[1.0, 2.0]);
    assert!(retained.into_host().is_ok());
}

#[test]
fn promotion_adopts_the_same_allocation_without_copying() {
    let host = TypedTensor::<f64, DynRank, Host>::from_host_vec_col_major(vec![2], vec![1.0, 2.0])
        .unwrap();
    let pointer = host.as_slice().as_ptr();
    let gpu: TypedTensor<f64, DynRank, Gpu> = host.promote().unwrap();
    assert_eq!(gpu.host_data().unwrap().as_ptr(), pointer);
    assert_eq!(gpu.host_data().unwrap(), &[1.0, 2.0]);
    assert!(!gpu.is_backend_buffer());
    // Read mapping still reaches the promoted host root.
    let dynamic = gpu.into_dynamic();
    assert_eq!(&dynamic.map_read().unwrap()[..], &[1.0, 2.0]);
    // A group-backed owner does not narrow back to plain host storage.
    assert!(dynamic.into_host().is_err());
}

#[test]
fn union_write_mapping_publishes_through_the_owning_guard() {
    let mut dynamic = TypedTensor::<f64>::from_vec_col_major(vec![2], vec![0.0, 0.0]).unwrap();
    dynamic
        .map_write()
        .unwrap()
        .copy_from_slice(&[5.0, 6.0])
        .unwrap();
    assert_eq!(dynamic.host_data().unwrap(), &[5.0, 6.0]);

    let handle = BackendStorageHandle::<f64>::new_with_len(12, 2);
    let device = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(handle)),
        Placement::default(),
    )
    .unwrap();
    assert!(device.map_read().is_err());
}

#[test]
fn rank_conversion_failure_retains_the_owner() {
    let tensor =
        TypedTensor::<f64>::from_vec_col_major(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
    let Err(failure) = tensor.try_into_rank::<3>() else {
        panic!("a rank-2 tensor is not rank 3");
    };
    assert!(!failure.error().to_string().is_empty());
    let retained = failure.into_owner();
    assert_eq!(retained.shape(), &[2, 3]);
    assert_eq!(
        retained.host_data().unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    // The retained owner still converts when the rank does match.
    assert_eq!(retained.try_into_rank::<2>().unwrap().shape(), &[2, 3]);
}

#[test]
fn dtype_recovery_failure_retains_the_erased_tensor() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f32, 2.0]).unwrap();
    let Err(failure) = tensor.into_typed::<f64>() else {
        panic!("an f32 tensor is not f64");
    };
    assert!(!failure.error().to_string().is_empty());
    let retained = failure.into_owner();
    assert_eq!(retained.dtype(), DType::F32);
    assert_eq!(retained.shape(), &[2]);
    assert_eq!(retained.as_slice::<f32>().unwrap(), &[1.0, 2.0]);
}

#[test]
fn promotion_keeps_the_pooled_return_target() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Weak};

    #[derive(Debug)]
    struct Count(AtomicUsize);
    impl tenferro_tensor::HostBufferRecycler<f64> for Count {
        fn recycle(&self, _data: Vec<f64>) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    let counter = Arc::new(Count(AtomicUsize::new(0)));
    let recycler: Arc<dyn tenferro_tensor::HostBufferRecycler<f64>> = counter.clone();
    let weak: Weak<dyn tenferro_tensor::HostBufferRecycler<f64>> = Arc::downgrade(&recycler);
    let pooled =
        TypedTensor::<f64>::from_vec_col_major_with_recycler(vec![2], vec![1.0, 2.0], weak)
            .unwrap();

    let host = pooled.into_host().unwrap();
    let gpu = host.promote().unwrap();
    assert_eq!(gpu.host_data().unwrap(), &[1.0, 2.0]);
    assert_eq!(
        counter.0.load(Ordering::SeqCst),
        0,
        "nothing recycles while owned"
    );
    drop(gpu);
    assert_eq!(
        counter.0.load(Ordering::SeqCst),
        1,
        "promotion must keep the pooled return target on the group root"
    );
}

#[test]
fn host_export_failure_retains_the_owner() {
    let handle = BackendStorageHandle::<f64>::new_with_len(21, 2);
    let device = TypedTensor::<f64>::from_buffer_col_major(
        vec![2],
        StorageBuffer::Backend(Box::new(handle)),
        Placement::default(),
    )
    .unwrap();

    let Err(failure) = device.into_host_vec() else {
        panic!("a device allocation has no host vector");
    };
    assert!(!failure.error().to_string().is_empty());
    let retained = failure.into_owner();
    assert_eq!(retained.shape(), &[2]);
    assert!(retained.backend_buffer().is_some());

    let Err(failure) = retained.into_vec_col_major() else {
        panic!("a device allocation has no host vector");
    };
    let retained = failure.into_owner();
    assert_eq!(retained.shape(), &[2]);

    let Err(failure) = retained.into_parts() else {
        panic!("a device allocation extracts no host storage");
    };
    let retained = failure.into_owner();
    assert!(retained.backend_buffer().is_some());
    // The retained device owner still narrows to the group-backed representation.
    assert!(retained.into_gpu().is_ok());
}

#[test]
fn erased_host_export_dtype_mismatch_retains_the_tensor() {
    let tensor = Tensor::from_vec_col_major(vec![2], vec![1.0_f64, 2.0]).unwrap();
    let Err(failure) = tensor.into_vec_col_major::<f32>() else {
        panic!("an f64 tensor is not f32");
    };
    assert!(!failure.error().to_string().is_empty());
    let retained = failure.into_owner();
    assert_eq!(retained.dtype(), DType::F64);
    assert_eq!(retained.as_slice::<f64>().unwrap(), &[1.0, 2.0]);
}
