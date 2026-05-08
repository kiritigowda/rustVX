//! OpenVX Vision Kernels - Complete Implementation
//!
//! This crate implements all vision conformant kernels from OpenVX 1.3.1
//! including color conversions, filter operations, gradient operations,
//! image arithmetic, statistical operations, geometric operations,
//! optical flow, feature detection, and object detection.

use openvx_core::{Context, Referenceable, VxKernel, VxResult};

pub mod arithmetic;
pub mod color;
pub mod features;
pub mod filter;
pub mod geometric;
pub mod gradient;
pub mod morphology;
pub mod object_detection;
pub mod optical_flow;
pub mod statistics;
pub mod utils;

// Kernel enum constants and registration
pub mod kernel_enums;
pub mod register;

// SIMD optimized modules (conditionally compiled)
pub mod simd_utils;

#[cfg(feature = "simd")]
pub mod arithmetic_simd;
#[cfg(feature = "simd")]
pub mod color_simd;
#[cfg(feature = "simd")]
pub mod filter_simd;

#[cfg(feature = "parallel")]
pub mod parallel;

// Platform-specific SIMD modules
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub mod aarch64_simd;
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub mod x86_64_simd;

/// Core trait for all vision kernels
pub trait Kernel: Send + Sync {
    fn get_name(&self) -> &str;
    fn get_enum(&self) -> VxKernel;
    fn validate(&self, params: &[&dyn Referenceable]) -> VxResult<()>;
    fn execute(&self, params: &[&dyn Referenceable], context: &Context) -> VxResult<()>;
}

/// Register all vision kernels with the context
pub fn register_all_kernels(context: &Context) -> VxResult<()> {
    // First, register vision kernels in the global KERNELS registry
    // This ensures they are accessible via vxGetKernelByName/vxQueryKernel
    register::register_vision_kernels_in_global_registry();

    // Color conversions
    context.register_kernel(Box::new(color::ColorConvertKernel))?;
    context.register_kernel(Box::new(color::ChannelExtractKernel))?;
    context.register_kernel(Box::new(color::ChannelCombineKernel))?;

    // Filter operations
    context.register_kernel(Box::new(filter::ConvolveKernel))?;
    context.register_kernel(Box::new(filter::Gaussian3x3Kernel))?;
    context.register_kernel(Box::new(filter::Gaussian5x5Kernel))?;
    context.register_kernel(Box::new(filter::Median3x3Kernel))?;
    context.register_kernel(Box::new(filter::Box3x3Kernel))?;

    // Morphology
    context.register_kernel(Box::new(morphology::Dilate3x3Kernel))?;
    context.register_kernel(Box::new(morphology::Erode3x3Kernel))?;

    // Gradient operations
    context.register_kernel(Box::new(gradient::Sobel3x3Kernel))?;
    context.register_kernel(Box::new(gradient::MagnitudeKernel))?;
    context.register_kernel(Box::new(gradient::PhaseKernel))?;

    // Image arithmetic
    context.register_kernel(Box::new(arithmetic::AddKernel))?;
    context.register_kernel(Box::new(arithmetic::SubtractKernel))?;
    context.register_kernel(Box::new(arithmetic::MultiplyKernel))?;
    context.register_kernel(Box::new(arithmetic::WeightedAverageKernel))?;

    // Enhanced Vision: pixel-wise min/max
    context.register_kernel(Box::new(arithmetic::MinKernel))?;
    context.register_kernel(Box::new(arithmetic::MaxKernel))?;

    // Bitwise logical operations
    context.register_kernel(Box::new(arithmetic::AndKernel))?;
    context.register_kernel(Box::new(arithmetic::OrKernel))?;
    context.register_kernel(Box::new(arithmetic::XorKernel))?;
    context.register_kernel(Box::new(arithmetic::NotKernel))?;

    // Statistical operations
    context.register_kernel(Box::new(statistics::MinMaxLocKernel))?;
    context.register_kernel(Box::new(statistics::MeanStdDevKernel))?;
    context.register_kernel(Box::new(statistics::HistogramKernel))?;
    context.register_kernel(Box::new(statistics::EqualizeHistogramKernel))?;
    context.register_kernel(Box::new(statistics::IntegralImageKernel))?;

    // Geometric operations
    context.register_kernel(Box::new(geometric::ScaleImageKernel))?;
    context.register_kernel(Box::new(geometric::WarpAffineKernel))?;
    context.register_kernel(Box::new(geometric::WarpPerspectiveKernel))?;

    // Optical flow
    context.register_kernel(Box::new(optical_flow::OpticalFlowPyrLKKernel))?;

    // Feature detection
    context.register_kernel(Box::new(features::HarrisCornersKernel))?;
    context.register_kernel(Box::new(features::FASTCornersKernel))?;

    // Object detection
    context.register_kernel(Box::new(object_detection::CannyEdgeDetectorKernel))?;
    context.register_kernel(Box::new(object_detection::HoughLinesPKernel))?;
    context.register_kernel(Box::new(object_detection::ThresholdKernel::new()))?;

    Ok(())
}

pub use arithmetic::{
    AddKernel, AndKernel, MaxKernel, MinKernel, MultiplyKernel, NotKernel, OrKernel,
    SubtractKernel, WeightedAverageKernel, XorKernel,
};
pub use color::{ChannelCombineKernel, ChannelExtractKernel, ColorConvertKernel};
pub use features::{FASTCornersKernel, HarrisCornersKernel};
pub use filter::{
    Box3x3Kernel, ConvolveKernel, Gaussian3x3Kernel, Gaussian5x5Kernel, Median3x3Kernel,
};
pub use geometric::{ScaleImageKernel, WarpAffineKernel, WarpPerspectiveKernel};
pub use gradient::{MagnitudeKernel, PhaseKernel, Sobel3x3Kernel};
pub use morphology::{Dilate3x3Kernel, Erode3x3Kernel};
pub use object_detection::{
    threshold_binary, threshold_range, CannyEdgeDetectorKernel, HoughLinesPKernel, ThresholdKernel,
    ThresholdType,
};
pub use optical_flow::OpticalFlowPyrLKKernel;
pub use statistics::{
    EqualizeHistogramKernel, HistogramKernel, IntegralImageKernel, MeanStdDevKernel,
    MinMaxLocKernel,
};

// Export utility functions
pub use utils::*;
