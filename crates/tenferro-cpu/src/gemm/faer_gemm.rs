use num_complex::{Complex32, Complex64};

use crate::provider::CpuExecutionContext;

pub(crate) trait FaerGemm: Sized {
    #[allow(clippy::too_many_arguments, dead_code)]
    unsafe fn strided_gemm(
        ctx: &CpuExecutionContext<'_>,
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    ) {
        unsafe {
            Self::strided_gemm_with_conj(
                ctx, alpha, a_ptr, m, k, a_rs, a_cs, false, b_ptr, n, b_rs, b_cs, false, beta,
                c_ptr, c_rs, c_cs,
            )
        }
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    unsafe fn strided_gemm_with_conj(
        ctx: &CpuExecutionContext<'_>,
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        conj_a: bool,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        conj_b: bool,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    ) {
        unsafe {
            Self::strided_gemm_with_conj_par(
                ctx,
                ctx.faer_parallelism(),
                alpha,
                a_ptr,
                m,
                k,
                a_rs,
                a_cs,
                conj_a,
                b_ptr,
                n,
                b_rs,
                b_cs,
                conj_b,
                beta,
                c_ptr,
                c_rs,
                c_cs,
            )
        }
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn strided_gemm_with_conj_par(
        ctx: &CpuExecutionContext<'_>,
        par: faer::Par,
        alpha: Self,
        a_ptr: *const Self,
        m: usize,
        k: usize,
        a_rs: isize,
        a_cs: isize,
        conj_a: bool,
        b_ptr: *const Self,
        n: usize,
        b_rs: isize,
        b_cs: isize,
        conj_b: bool,
        beta: Self,
        c_ptr: *mut Self,
        c_rs: isize,
        c_cs: isize,
    );
}

/// Real multiply-adds below which faer's parallel matmul is slower than a
/// sequential one. Measured on an AMD EPYC host with faer 0.24: at 64^3 the
/// parallel call took 19.6/21.3/31.0 us at 4/8/16 threads against 14.0 us
/// sequential, while at 128^3 it won (55.7 us at 4 threads against 96.4 us).
/// Complex elements count four real multiply-adds each.
const FAER_PARALLEL_MIN_MULADDS: usize = 1 << 20;

/// Keep a small GEMM on the calling thread even inside a multi-threaded
/// context, so a multi-threaded backend is never slower than a single-threaded
/// one on small products (PERFORMANCE_TIPS CPU threading contract).
pub(super) fn small_gemm_parallelism<T: 'static>(
    par: faer::Par,
    m: usize,
    n: usize,
    k: usize,
) -> faer::Par {
    if matches!(par, faer::Par::Seq) {
        return par;
    }
    let complex = std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex64>()
        || std::any::TypeId::of::<T>() == std::any::TypeId::of::<Complex32>();
    let weight = if complex { 4 } else { 1 };
    let muladds = m.saturating_mul(n).saturating_mul(k).saturating_mul(weight);
    if muladds < FAER_PARALLEL_MIN_MULADDS {
        faer::Par::Seq
    } else {
        par
    }
}

macro_rules! impl_faer_gemm {
    ($ty:ty) => {
        impl FaerGemm for $ty {
            unsafe fn strided_gemm_with_conj_par(
                ctx: &CpuExecutionContext<'_>,
                par: faer::Par,
                alpha: $ty,
                a_ptr: *const $ty,
                m: usize,
                k: usize,
                a_rs: isize,
                a_cs: isize,
                conj_a: bool,
                b_ptr: *const $ty,
                n: usize,
                b_rs: isize,
                b_cs: isize,
                conj_b: bool,
                beta: $ty,
                c_ptr: *mut $ty,
                c_rs: isize,
                c_cs: isize,
            ) {
                let _ = ctx;
                use faer::{Accum, Conj, MatMut, MatRef};
                let a_rs = super::normalize_singleton_stride(a_rs, m, k);
                let a_cs = super::normalize_singleton_stride(a_cs, k, m);
                let b_rs = super::normalize_singleton_stride(b_rs, k, n);
                let b_cs = super::normalize_singleton_stride(b_cs, n, k);
                let c_rs = super::normalize_singleton_stride(c_rs, m, 1);
                let c_cs = super::normalize_singleton_stride(c_cs, n, m);
                // SAFETY: callers pass dense tensor storage plus validated GEMM
                // dimensions/strides; singleton strides are normalized above.
                let a_mat = MatRef::<$ty>::from_raw_parts(a_ptr, m, k, a_rs, a_cs);
                let b_mat = MatRef::<$ty>::from_raw_parts(b_ptr, k, n, b_rs, b_cs);
                let zero = <$ty as num_traits::Zero>::zero();
                let one = <$ty as num_traits::One>::one();
                let accum = if beta == zero {
                    Accum::Replace
                } else {
                    if beta != one {
                        // SAFETY: beta != 0 callers must pass initialized C; pooled dot_general uses beta = 0.
                        let mut col_off = 0isize;
                        for _ in 0..n {
                            let mut off = col_off;
                            for _ in 0..m {
                                *c_ptr.offset(off) *= beta;
                                off += c_rs;
                            }
                            col_off += c_cs;
                        }
                    }
                    Accum::Add
                };
                // SAFETY: `c_ptr` points at the caller-owned output buffer for
                // an m x n dense result; beta=0 paths may receive uninitialized
                // storage because faer uses Accum::Replace.
                let mut c_mat = MatMut::<$ty>::from_raw_parts_mut(c_ptr, m, n, c_rs, c_cs);
                let conj_a = if conj_a { Conj::Yes } else { Conj::No };
                let conj_b = if conj_b { Conj::Yes } else { Conj::No };
                let par = small_gemm_parallelism::<$ty>(par, m, n, k);
                faer::linalg::matmul::matmul_with_conj(
                    &mut c_mat, accum, &a_mat, conj_a, &b_mat, conj_b, alpha, par,
                );
            }
        }
    };
}

impl_faer_gemm!(f64);
impl_faer_gemm!(f32);
impl_faer_gemm!(Complex64);
impl_faer_gemm!(Complex32);
