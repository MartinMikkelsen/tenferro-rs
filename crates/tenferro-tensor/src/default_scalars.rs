//! The crate's default scalar set as a dtype-erased host value.
//!
//! [`DefaultScalars`] is tenferro's own [`ScalarSet`]: one value holding a
//! host tensor of any preset dtype. Each member is the canonical
//! `TypedTensor<T, DynRank, Host>` (views: `TypedTensorView<'a, T, DynRank,
//! Host>`), so the set adds dtype erasure without a second owner or view
//! family. `tenferro-tensor-core` keeps the scalar tags and promotion facts.

use num_complex::{Complex32, Complex64};
use tenferro_tensor_core::{promote_in_set, DType, DynRank};

use crate::{Host, ScalarSet, StridedSliceSpec, TensorScalar, TypedTensor, TypedTensorView};

fn dtype_mismatch(op: &'static str, expected: DType, actual: DType) -> crate::Error {
    crate::Error::validation(
        op,
        tenferro_tensor_core::ValidationError::DTypeMismatch { expected, actual },
    )
}

/// Dynamic host tensor over the crate's preset scalar set.
///
/// The payload is a private inline value: a host tensor is one move and a copy
/// allocates nothing, and the preset variants are not part of the public
/// surface. Construct through [`DefaultScalars::from_vec_col_major`] and read
/// through [`DefaultScalars::as_slice`], [`DefaultScalars::as_mut_slice`], or
/// [`DefaultScalars::into_vec_col_major`].
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DefaultScalars, DType};
/// use tenferro_tensor::ScalarSet;
///
/// let value = DefaultScalars::from_vec_col_major(vec![1], vec![7_i32])?;
/// assert_eq!(value.tag(), DType::I32);
/// assert_eq!(value.as_slice::<i32>()?, &[7]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct DefaultScalars {
    value: DefaultScalarsValue,
}

/// Private payload of [`DefaultScalars`].
///
/// The variants are not part of the public surface: construct through
/// [`DefaultScalars::from_vec_col_major`] and read through
/// [`DefaultScalars::as_slice`].
#[derive(Clone, Debug)]
pub(crate) enum DefaultScalarsValue {
    F32(TypedTensor<f32, DynRank, Host>),
    F64(TypedTensor<f64, DynRank, Host>),
    I32(TypedTensor<i32, DynRank, Host>),
    I64(TypedTensor<i64, DynRank, Host>),
    Bool(TypedTensor<bool, DynRank, Host>),
    C32(TypedTensor<Complex32, DynRank, Host>),
    C64(TypedTensor<Complex64, DynRank, Host>),
}

impl ScalarSet for DefaultScalarsValue {
    type Tag = DType;

    const TAGS: &'static [Self::Tag] = DType::TAGS;

    fn tag(&self) -> Self::Tag {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    fn promote(lhs: Self::Tag, rhs: Self::Tag) -> Self::Tag {
        // An externally defined member promotes to itself, because tenferro
        // declares no facts that relate it to one of its own members.
        if matches!(lhs, DType::External(_)) {
            return lhs;
        }
        if matches!(rhs, DType::External(_)) {
            return rhs;
        }
        promote_in_set(DType::TAGS, DType::SPECS, lhs, rhs)
    }
}

impl DefaultScalarsValue {
    /// Return the tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![false])?;
    /// assert_eq!(tensor.dtype(), DType::Bool);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Borrow the tensor shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![2], vec![1_i32, 2])?;
    /// assert_eq!(tensor.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(t) => t.shape(),
            Self::F64(t) => t.shape(),
            Self::I32(t) => t.shape(),
            Self::I64(t) => t.shape(),
            Self::Bool(t) => t.shape(),
            Self::C32(t) => t.shape(),
            Self::C64(t) => t.shape(),
        }
    }

    /// Return the tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(tensor.rank(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return whether the tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(tensor.is_empty());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            Self::F32(t) => t.shape().contains(&0),
            Self::F64(t) => t.shape().contains(&0),
            Self::I32(t) => t.shape().contains(&0),
            Self::I64(t) => t.shape().contains(&0),
            Self::Bool(t) => t.shape().contains(&0),
            Self::C32(t) => t.shape().contains(&0),
            Self::C64(t) => t.shape().contains(&0),
        }
    }

    /// Borrow this tensor as a dynamic zero-offset view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(tensor.as_view().dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_view(&self) -> DefaultScalarsView<'_> {
        match self {
            Self::F32(t) => DefaultScalarsView::F32(t.as_view()),
            Self::F64(t) => DefaultScalarsView::F64(t.as_view()),
            Self::I32(t) => DefaultScalarsView::I32(t.as_view()),
            Self::I64(t) => DefaultScalarsView::I64(t.as_view()),
            Self::Bool(t) => DefaultScalarsView::Bool(t.as_view()),
            Self::C32(t) => DefaultScalarsView::C32(t.as_view()),
            Self::C64(t) => DefaultScalarsView::C64(t.as_view()),
        }
    }
}

impl DefaultScalars {
    /// Create a dynamic tensor from a column-major host buffer.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![2.0_f32])?;
    /// assert_eq!(tensor.dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error carrying
    /// [`tenferro_tensor_core::ValidationError::ShapeDataLengthMismatch`] when the shape product
    /// differs from `data.len()`, or [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// when validating the shape overflows.
    pub fn from_vec_col_major<T: TensorScalar>(
        shape: impl Into<Vec<usize>>,
        data: Vec<T>,
    ) -> crate::Result<Self> {
        T::into_default_scalars(shape.into(), data)
    }

    /// Return the tensor dtype tag.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![false])?;
    /// assert_eq!(tensor.dtype(), DType::Bool);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        self.value.dtype()
    }

    /// Borrow the tensor shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![2], vec![1_i32, 2])?;
    /// assert_eq!(tensor.shape(), &[2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        self.value.shape()
    }

    /// Return the tensor rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(tensor.rank(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.value.rank()
    }

    /// Return whether the tensor has zero elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(tensor.is_empty());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        self.value.is_empty()
    }

    /// Borrow the typed host slice when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// assert_eq!(tensor.as_slice::<f64>()?, &[3.0]);
    /// assert!(tensor.as_slice::<f32>().is_err());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a dtype mismatch when `T`
    /// does not match the tensor's runtime dtype.
    pub fn as_slice<T: TensorScalar>(&self) -> crate::Result<&[T]> {
        let actual = self.dtype();
        T::default_scalars_slice(self)
            .ok_or_else(|| dtype_mismatch("DefaultScalars::as_slice", T::dtype(), actual))
    }

    /// Mutably borrow the typed host slice when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let mut tensor = DefaultScalars::from_vec_col_major(vec![1], vec![3.0_f64])?;
    /// tensor.as_mut_slice::<f64>()?[0] = 4.0;
    /// assert_eq!(tensor.as_slice::<f64>()?, &[4.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a dtype mismatch when `T`
    /// does not match the tensor's runtime dtype.
    pub fn as_mut_slice<T: TensorScalar>(&mut self) -> crate::Result<&mut [T]> {
        let actual = self.dtype();
        T::default_scalars_slice_mut(self)
            .ok_or_else(|| dtype_mismatch("DefaultScalars::as_mut_slice", T::dtype(), actual))
    }

    /// Borrow this tensor as a dynamic zero-offset view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f32])?;
    /// assert_eq!(tensor.as_view().dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn as_view(&self) -> DefaultScalarsView<'_> {
        self.value.as_view()
    }

    /// Consume this tensor and return typed column-major data when the dtype matches.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![2], vec![1.0_f64, 2.0])?;
    /// let (shape, data) = tensor.into_vec_col_major::<f64>()?;
    /// assert_eq!(shape, vec![2]);
    /// assert_eq!(data, vec![1.0, 2.0]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] with a dtype mismatch when `T`
    /// does not match the tensor's runtime dtype.
    pub fn into_vec_col_major<T: TensorScalar>(self) -> crate::Result<(Vec<usize>, Vec<T>)> {
        let actual = self.dtype();
        T::from_default_scalars(self)
            .map(TypedTensor::<T, DynRank, Host>::into_vec_col_major)
            .ok_or_else(|| dtype_mismatch("DefaultScalars::into_vec_col_major", T::dtype(), actual))
    }

    pub(crate) fn from_payload(value: DefaultScalarsValue) -> Self {
        Self { value }
    }

    pub(crate) fn payload(&self) -> &DefaultScalarsValue {
        &self.value
    }

    pub(crate) fn payload_mut(&mut self) -> &mut DefaultScalarsValue {
        &mut self.value
    }

    pub(crate) fn into_payload(self) -> DefaultScalarsValue {
        self.value
    }
}

impl ScalarSet for DefaultScalars {
    type Tag = DType;

    const TAGS: &'static [Self::Tag] = <DefaultScalarsValue as ScalarSet>::TAGS;

    fn tag(&self) -> Self::Tag {
        self.value.tag()
    }

    fn promote(lhs: Self::Tag, rhs: Self::Tag) -> Self::Tag {
        <DefaultScalarsValue as ScalarSet>::promote(lhs, rhs)
    }
}

/// Dynamic borrowed host tensor view.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DefaultScalars, DType};
///
/// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![true])?;
/// let view = tensor.as_view();
/// assert_eq!(view.dtype(), DType::Bool);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
///
/// ```compile_fail
/// # use tenferro_tensor::DefaultScalars;
/// # let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f64]).unwrap();
/// let a = tensor.as_view();
/// let b = tensor.as_view();
/// let _ = a == b;
/// ```
#[derive(Clone, Debug)]
pub enum DefaultScalarsView<'a> {
    F32(TypedTensorView<'a, f32, DynRank, Host>),
    F64(TypedTensorView<'a, f64, DynRank, Host>),
    I32(TypedTensorView<'a, i32, DynRank, Host>),
    I64(TypedTensorView<'a, i64, DynRank, Host>),
    Bool(TypedTensorView<'a, bool, DynRank, Host>),
    C32(TypedTensorView<'a, Complex32, DynRank, Host>),
    C64(TypedTensorView<'a, Complex64, DynRank, Host>),
}

macro_rules! impl_dynamic_view {
    ($self:ident, $method:ident($($arg:ident),*) => $inner:ident) => {
        match $self {
            DefaultScalarsView::F32(view) => DefaultScalarsView::F32(view.$method($($arg),*)?),
            DefaultScalarsView::F64(view) => DefaultScalarsView::F64(view.$method($($arg),*)?),
            DefaultScalarsView::I32(view) => DefaultScalarsView::I32(view.$method($($arg),*)?),
            DefaultScalarsView::I64(view) => DefaultScalarsView::I64(view.$method($($arg),*)?),
            DefaultScalarsView::Bool(view) => DefaultScalarsView::Bool(view.$method($($arg),*)?),
            DefaultScalarsView::C32(view) => DefaultScalarsView::C32(view.$method($($arg),*)?),
            DefaultScalarsView::C64(view) => DefaultScalarsView::C64(view.$method($($arg),*)?),
        }
    };
}

impl<'a> DefaultScalarsView<'a> {
    /// Return this view's dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f32])?;
    /// assert_eq!(tensor.as_view().dtype(), DType::F32);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::F32(_) => DType::F32,
            Self::F64(_) => DType::F64,
            Self::I32(_) => DType::I32,
            Self::I64(_) => DType::I64,
            Self::Bool(_) => DType::Bool,
            Self::C32(_) => DType::C32,
            Self::C64(_) => DType::C64,
        }
    }

    /// Borrow this view's shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert_eq!(tensor.as_view().shape(), &[1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(view) => view.shape(),
            Self::F64(view) => view.shape(),
            Self::I32(view) => view.shape(),
            Self::I64(view) => view.shape(),
            Self::Bool(view) => view.shape(),
            Self::C32(view) => view.shape(),
            Self::C64(view) => view.shape(),
        }
    }

    /// Return the view rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(tensor.as_view().rank(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return whether this view has zero logical elements.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(tensor.as_view().is_empty());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            Self::F32(view) => view.shape().contains(&0),
            Self::F64(view) => view.shape().contains(&0),
            Self::I32(view) => view.shape().contains(&0),
            Self::I64(view) => view.shape().contains(&0),
            Self::Bool(view) => view.shape().contains(&0),
            Self::C32(view) => view.shape().contains(&0),
            Self::C64(view) => view.shape().contains(&0),
        }
    }

    /// Return a metadata-only reshape of this dynamic view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![4], vec![1_i32, 2, 3, 4])?;
    /// assert_eq!(tensor.as_view().reshape_view(vec![2, 2])?.shape(), &[2, 2]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error carrying [`tenferro_tensor_core::ValidationError::ShapeMismatch`]
    /// when the element counts differ,
    /// [`tenferro_tensor_core::ValidationError::NonContiguousViewAsSlice`] when the view's strides
    /// cannot express the new shape, or [`tenferro_tensor_core::ValidationError::IntegerOverflow`]
    /// when shape arithmetic overflows.
    pub fn reshape_view(&self, shape: impl AsRef<[usize]>) -> crate::Result<Self> {
        let shape = shape.as_ref();
        Ok(impl_dynamic_view!(self, reshape_view(shape) => view))
    }

    /// Return a metadata-only transposed dynamic view with axes in the requested order.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::DefaultScalars;
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1, 2], vec![1_i64, 2])?;
    /// assert_eq!(tensor.as_view().transpose_view(&[1, 0])?.shape(), &[2, 1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns a validation error carrying
    /// [`tenferro_tensor_core::ValidationError::InvalidPermutationLength`],
    /// [`tenferro_tensor_core::ValidationError::AxisOutOfBounds`] or
    /// [`tenferro_tensor_core::ValidationError::DuplicateAxis`] when `axes` is not a permutation of
    /// the view rank.
    pub fn transpose_view(&self, axes: &[usize]) -> crate::Result<Self> {
        Ok(impl_dynamic_view!(self, transpose_view(axes) => view))
    }

    /// Return a metadata-only positive-step slice of this dynamic view.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, StridedSliceSpec};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![3], vec![1_i64, 2, 3])?;
    /// assert_eq!(
    ///     tensor.as_view().slice_view(&[StridedSliceSpec::new(1, Some(3), 1)])?.shape(),
    ///     &[2],
    /// );
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`crate::Error::Validation`] for a rank mismatch or invalid
    /// slice bounds or step.
    pub fn slice_view(&self, spec: &[StridedSliceSpec]) -> crate::Result<Self> {
        Ok(impl_dynamic_view!(self, slice_view(spec) => view))
    }
}

/// Core-neutral tensor input reference.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DefaultScalars, DefaultScalarsRef};
///
/// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f32])?;
/// let reference = DefaultScalarsRef::Tensor(&tensor);
/// assert_eq!(reference.shape(), &[1]);
/// # Ok::<(), tenferro_tensor::Error>(())
/// ```
// The view variant holds a canonical `TypedTensorView` inline so a borrowed
// input costs no allocation; boxing it would allocate on every reference.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum DefaultScalarsRef<'a> {
    Tensor(&'a DefaultScalars),
    View(DefaultScalarsView<'a>),
}

impl<'a> DefaultScalarsRef<'a> {
    /// Return the referenced dtype.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DefaultScalarsRef, DType};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(DefaultScalarsRef::Tensor(&tensor).dtype(), DType::I64);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn dtype(&self) -> DType {
        match self {
            Self::Tensor(tensor) => tensor.dtype(),
            Self::View(view) => view.dtype(),
        }
    }

    /// Borrow the referenced shape.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DefaultScalarsRef};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1], vec![1_i64])?;
    /// assert_eq!(DefaultScalarsRef::Tensor(&tensor).shape(), &[1]);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Tensor(tensor) => tensor.shape(),
            Self::View(view) => view.shape(),
        }
    }

    /// Return the referenced rank.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DefaultScalarsRef};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![1, 1], vec![1_i64])?;
    /// assert_eq!(DefaultScalarsRef::Tensor(&tensor).rank(), 2);
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Return whether the referenced tensor/view is empty.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DefaultScalarsRef};
    ///
    /// let tensor = DefaultScalars::from_vec_col_major(vec![0], Vec::<f64>::new())?;
    /// assert!(DefaultScalarsRef::Tensor(&tensor).is_empty());
    /// # Ok::<(), tenferro_tensor::Error>(())
    /// ```
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Tensor(tensor) => tensor.is_empty(),
            Self::View(view) => view.shape().contains(&0),
        }
    }
}
