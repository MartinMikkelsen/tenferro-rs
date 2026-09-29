//! The import and construction forms a downstream crate uses for the default
//! scalar set.
//!
//! The value type is opaque: the preset variants are private, so a downstream
//! crate reaches the payload through the public constructor and the typed
//! accessors. `DefaultScalars` is the crate's public name for the value type;
//! the scalar-set traits stay in `tenferro-tensor-core`.

mod direct_path {
    use tenferro_tensor::{DType, DefaultScalars};

    #[test]
    fn the_canonical_path_names_the_value_type() {
        let value = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
        assert_eq!(value.dtype(), DType::F64);
        assert_eq!(value.as_slice::<f64>().unwrap(), &[1.0]);
    }
}

mod tagged_path {
    use tenferro_tensor::ScalarSet;
    use tenferro_tensor::{DType, DefaultScalars};

    #[test]
    fn the_tag_is_the_dtype() {
        let value = DefaultScalars::from_vec_col_major(vec![1], vec![7_i32]).unwrap();
        assert_eq!(value.tag(), DType::I32);
    }
}
