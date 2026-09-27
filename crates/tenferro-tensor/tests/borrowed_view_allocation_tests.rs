use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::hint::black_box;

use num_complex::Complex64;
use tenferro_tensor::{Rank, TypedTensor, TypedTensorView, TypedTensorViewMut};

struct CountingAllocator;

thread_local! {
    static COUNTING: Cell<bool> = const { Cell::new(false) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        // SAFETY: this allocator forwards the unchanged layout to System.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: ptr and layout came from the corresponding System allocation.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if COUNTING.get() {
            ALLOCATIONS.set(ALLOCATIONS.get() + 1);
        }
        // SAFETY: ptr and layout came from System, and new_size is forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn count_allocations(op: impl FnOnce()) -> usize {
    ALLOCATIONS.set(0);
    COUNTING.set(true);
    op();
    COUNTING.set(false);
    ALLOCATIONS.get()
}

#[test]
fn small_dynamic_borrowed_view_metadata_stays_inline() {
    let data = [0_i32; 6];
    let read_allocations = count_allocations(|| {
        let view = TypedTensorView::from_slice([2, 3], [1, 2], 0, &data).unwrap();
        black_box(view);
    });

    let mut data = [0_i32; 6];
    let write_allocations = count_allocations(|| {
        let view = TypedTensorViewMut::from_slice([2, 3], [1, 2], 0, &mut data).unwrap();
        black_box(view);
    });

    assert_eq!(read_allocations, 0);
    assert_eq!(write_allocations, 0);
}

#[test]
fn static_rank_borrowed_erasure_does_not_allocate_or_promote_storage() {
    let data = [1.0_f64, 2.0, 3.0, 4.0];
    let allocations = count_allocations(|| {
        let view =
            TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 2], [1, 2], 0, &data).unwrap();
        let read = view.into_tensor_read().unwrap();
        assert_eq!(read.as_slice::<f64>().unwrap(), &data);
        black_box(read);
    });
    assert_eq!(allocations, 0);
}

#[test]
fn ranked_owner_view_erasure_keeps_its_root_borrow() {
    let owner =
        TypedTensor::<f64, Rank<2>>::from_vec_col_major([2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let read = owner.as_view().into_tensor_read().unwrap();
    assert_eq!(read.as_slice::<f64>().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn small_complex_real_views_do_not_allocate_metadata() {
    let data = [Complex64::new(1.0, 2.0); 6];
    let immutable = count_allocations(|| {
        let view =
            TypedTensorView::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &data).unwrap();
        let real = view.as_real_view().unwrap();
        assert_eq!(real.shape(), &[2, 2, 3]);
        assert_eq!(real.as_slice().unwrap()[1], 2.0);
        black_box(real);
    });
    let mut data = [Complex64::new(1.0, 2.0); 6];
    let mutable = count_allocations(|| {
        let mut view =
            TypedTensorViewMut::<_, Rank<2>>::from_slice_ranked([2, 3], [1, 2], 0, &mut data)
                .unwrap();
        let mut real = view.as_real_view_mut().unwrap();
        real.host_storage_mut().unwrap()[1] = 4.0;
        black_box(real);
    });
    let dynamic = count_allocations(|| {
        let view = TypedTensorView::from_slice([2, 3], [1, 2], 0, &data).unwrap();
        black_box(view.as_real_view().unwrap());
    });
    let dynamic_mut = count_allocations(|| {
        let mut view = TypedTensorViewMut::from_slice([2, 3], [1, 2], 0, &mut data).unwrap();
        black_box(view.as_real_view_mut().unwrap());
    });
    assert_eq!(data[0].im, 4.0);
    assert_eq!(immutable, 0);
    assert_eq!(mutable, 0);
    assert_eq!(dynamic, 0);
    assert_eq!(dynamic_mut, 0);

    // Arbitrary rank may spill, but its layout must still be valid.
    let high_rank = TypedTensorView::from_slice([1; 8], [1; 8], 0, &data[..1]).unwrap();
    assert_eq!(
        high_rank.as_real_view().unwrap().shape(),
        &[2, 1, 1, 1, 1, 1, 1, 1, 1]
    );
}
