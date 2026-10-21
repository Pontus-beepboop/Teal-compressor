//! Small threading helper shared by the codec stages.

use crate::Result;

/// Runs `f(i)` for i in 0..n on the rayon pool; returns the first error. (`threads` is
/// kept for call-site symmetry; the global pool already uses every core.)
pub fn parallel_for<F>(n: usize, _threads: usize, f: F) -> Result<()>
where
    F: Fn(usize) -> Result<()> + Sync + Send,
{
    use rayon::prelude::*;
    (0..n).into_par_iter().try_for_each(f)
}

/// A raw pointer that may be shared across the threads of [`parallel_for`] when every
/// task writes a disjoint region.
#[derive(Clone, Copy)]
pub struct SyncPtr<T>(pub *mut T);
unsafe impl<T> Send for SyncPtr<T> {}
unsafe impl<T> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    /// # Safety
    /// The caller guarantees `[at, at + len)` is in bounds and not accessed by any other
    /// thread while the slice lives.
    #[inline]
    #[allow(clippy::mut_from_ref)] // the pointer, not &self, is what is borrowed
    pub unsafe fn slice(&self, at: usize, len: usize) -> &mut [T] {
        std::slice::from_raw_parts_mut(self.0.add(at), len)
    }
}
