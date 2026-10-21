//! GPU backends for the resize pipeline. Each returns None when unavailable.

use crate::quality::Geometry;
use crate::resize::Resizer;
use crate::Backend;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;

#[cfg(feature = "cuda")]
pub mod cuda;

pub fn create(backend: Backend, src: Geometry, dst: Geometry) -> Option<Box<dyn Resizer>> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if matches!(backend, Backend::Auto | Backend::Metal) {
        if let Some(r) = metal::MetalResizer::new(src, dst) {
            return Some(Box::new(r));
        }
    }
    #[cfg(feature = "cuda")]
    if matches!(backend, Backend::Auto | Backend::Cuda) {
        if let Some(r) = cuda::CudaResizer::new(src, dst) {
            return Some(Box::new(r));
        }
    }
    let _ = (backend, src, dst);
    None
}

/// Name of the GPU a backend would use, if any.
pub fn description(backend: Backend) -> Option<String> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if matches!(backend, Backend::Auto | Backend::Metal) {
        if let Some(d) = metal::device_name() {
            return Some(format!("Metal: {d}"));
        }
    }
    #[cfg(feature = "cuda")]
    if matches!(backend, Backend::Auto | Backend::Cuda) {
        if let Some(d) = cuda::device_name() {
            return Some(format!("CUDA: {d}"));
        }
    }
    let _ = backend;
    None
}
