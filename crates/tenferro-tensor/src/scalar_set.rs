//! Closed scalar sets.
//!
//! A scalar set is the closed list of scalar types one tensor value type can
//! carry. tenferro declares its own set once (as
//! [`DefaultScalars`](crate::DefaultScalars)) and gets the value enum, the tag,
//! and the membership query from that single declaration. A crate that needs a
//! different set declares it with [`define_scalar_set!`] in its own crate, and
//! its values implement this trait without touching tenferro's set.
//!
//! The trait and its declaration macro live with the host container they build
//! values from; the promotion facts themselves stay in `tenferro-tensor-core`.

/// A closed set of scalar types carried by one tensor value type.
///
/// A set is represented by one value type: the default set's payload is opaque
/// and a downstream set defines its own value enum, and [`ScalarSet::tag`]
/// reports which member a value currently holds.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{DefaultScalars, ScalarSet};
///
/// let value = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f64])?;
/// assert_eq!(value.tag(), tenferro_tensor_core::DType::F64);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
pub trait ScalarSet: Clone + core::fmt::Debug + 'static {
    /// Tag identifying one member of this set.
    type Tag: Copy + Eq + core::fmt::Debug + 'static;

    /// Tags of every member, in declaration order.
    const TAGS: &'static [Self::Tag];

    /// Tag of the member this value currently holds.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType, ScalarSet};
    ///
    /// let value = DefaultScalars::from_vec_col_major(vec![1], vec![1.0_f64])?;
    /// assert_eq!(value.tag(), DType::F64);
    /// # Ok::<(), tenferro_tensor_core::ValidationError>(())
    /// ```
    fn tag(&self) -> Self::Tag;

    /// Promote two members of this set to the member that represents both.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tenferro_tensor::{DefaultScalars, DType, ScalarSet};
    ///
    /// assert_eq!(
    ///     <DefaultScalars as ScalarSet>::promote(DType::I32, DType::F32),
    ///     DType::F64
    /// );
    /// ```
    fn promote(lhs: Self::Tag, rhs: Self::Tag) -> Self::Tag;
}

/// Define a closed scalar set: its tag type, its value enum, and its membership.
///
/// The declaration lists each member once. The macro emits the tag enum, the
/// value enum whose variants hold a
/// [`HostTensor`](tenferro_tensor::HostTensor) of the member type, and the
/// [`ScalarSet`] implementation. A downstream crate invokes this in its own
/// crate, so tenferro never needs to know the set.
///
/// # Examples
///
/// ```rust
/// use tenferro_tensor::{define_scalar_set, HostTensor, ScalarSet};
///
/// define_scalar_set! {
///     /// Tag for a two-member set.
///     pub enum PairTag {
///         /// Double precision.
///         F64 => f64 : Float 1 64,
///         /// Single precision.
///         F32 => f32 : Float 0 32,
///     }
///     /// Value enum for a two-member set.
///     pub enum Pair;
/// }
///
/// let value = Pair::F32(HostTensor::from_vec_col_major(vec![1], vec![1.0_f32])?);
/// assert_eq!(value.tag(), PairTag::F32);
/// assert_eq!(<Pair as ScalarSet>::TAGS, &[PairTag::F64, PairTag::F32]);
/// # Ok::<(), tenferro_tensor_core::ValidationError>(())
/// ```
#[macro_export]
macro_rules! define_scalar_set {
    (
        $(#[$tag_meta:meta])*
        $tag_vis:vis enum $tag:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident => $ty:ty : $kind:ident $level:literal $width:literal
            ),+ $(,)?
        }
        $(#[$set_meta:meta])*
        $set_vis:vis enum $set:ident;
        $( external $ext_variant:ident($ext_ty:ty); )?
    ) => {
        ::tenferro_tensor_core::define_scalar_tag! {
            $(#[$tag_meta])*
            $tag_vis enum $tag {
                $(
                    $(#[$variant_meta])*
                    $variant => $ty : $kind $level $width
                ),+
            }
            $( external $ext_variant($ext_ty); )?
        }

        $(#[$set_meta])*
        #[derive(Clone, Debug, PartialEq)]
        $set_vis enum $set {
            $(
                $(#[$variant_meta])*
                $variant($crate::HostTensor<$ty>),
            )+
        }

        impl $crate::ScalarSet for $set {
            type Tag = $tag;

            const TAGS: &'static [Self::Tag] = &[
                $(
                    $tag::$variant,
                )+
            ];

            fn tag(&self) -> Self::Tag {
                match self {
                    $(
                        $set::$variant(_) => $tag::$variant,
                    )+
                }
            }

            fn promote(lhs: Self::Tag, rhs: Self::Tag) -> Self::Tag {
                $(
                    if matches!(lhs, $tag::$ext_variant(_)) {
                        return lhs;
                    }
                    if matches!(rhs, $tag::$ext_variant(_)) {
                        return rhs;
                    }
                )?
                ::tenferro_tensor_core::promote_in_set(<$tag>::TAGS, <$tag>::SPECS, lhs, rhs)
            }
        }
    };
}
