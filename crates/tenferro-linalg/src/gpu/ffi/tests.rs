use super::cusolver::CudaDataType;

#[test]
fn cusolver_cuda_data_type_has_c_abi_integer_layout() {
    assert_eq!(
        std::mem::size_of::<CudaDataType>(),
        std::mem::size_of::<i32>()
    );
    assert_eq!(
        std::mem::align_of::<CudaDataType>(),
        std::mem::align_of::<i32>()
    );
}

#[test]
#[ignore = "requires the cuSOLVER and cuBLAS runtime libraries"]
fn vendor_libraries_are_loaded_once_per_process() {
    // Every handle shares one loaded library, so dropping backends never
    // unloads and reloads cuSOLVER or cuBLAS (#1924).
    let solver = super::cusolver::CusolverLibrary::load().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &solver,
        &super::cusolver::CusolverLibrary::load().unwrap()
    ));
    let blas = super::cusolver::CublasLibrary::load().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &blas,
        &super::cusolver::CublasLibrary::load().unwrap()
    ));
}
