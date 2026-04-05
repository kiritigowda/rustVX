//! OpenVX Vision Kernel Enum Constants
//!
//! These are the kernel enumeration values from the OpenVX 1.3.1 specification.
//! They are defined here so they can be used when registering vision kernels
//! in both the Rust context and the global C API KERNELS registry.

/// Vendor ID for Khronos (standard OpenVX kernels)
pub const VX_ID_KHRONOS: u32 = 0x000;

/// Library base for vision kernels
pub const VX_LIBRARY_KHR_BASE: u32 = 0x0;

/// Macro to compute kernel base: VX_KERNEL_BASE(vendor, lib) = ((vendor) << 20) | ((lib) << 12)
#[macro_export]
macro_rules! VX_KERNEL_BASE {
    ($vendor:expr, $lib:expr) => {
        (($vendor as i32) << 20) | (($lib as i32) << 12)
    };
}

/// Standard OpenVX vision kernel enum values
/// These align with the enum values defined in openvx-core/src/c_api.rs
pub const VX_KERNEL_COLOR_CONVERT: i32 = 0x00;
pub const VX_KERNEL_CHANNEL_EXTRACT: i32 = 0x01;
pub const VX_KERNEL_CHANNEL_COMBINE: i32 = 0x02;
pub const VX_KERNEL_SOBEL_3x3: i32 = 0x03;
pub const VX_KERNEL_MAGNITUDE: i32 = 0x04;
pub const VX_KERNEL_PHASE: i32 = 0x05;
pub const VX_KERNEL_SCALE_IMAGE: i32 = 0x06;
pub const VX_KERNEL_WARP_AFFINE: i32 = 0x07;
pub const VX_KERNEL_WARP_PERSPECTIVE: i32 = 0x08;
pub const VX_KERNEL_ADD: i32 = 0x09;
pub const VX_KERNEL_SUBTRACT: i32 = 0x0A;
pub const VX_KERNEL_MULTIPLY: i32 = 0x0B;
pub const VX_KERNEL_WEIGHTED_AVERAGE: i32 = 0x0C;
pub const VX_KERNEL_CONVOLVE: i32 = 0x0D;
pub const VX_KERNEL_GAUSSIAN_3x3: i32 = 0x0E;
pub const VX_KERNEL_MEDIAN_3x3: i32 = 0x0F;
pub const VX_KERNEL_SOBEL_5x5: i32 = 0x10;
pub const VX_KERNEL_BOX_3x3: i32 = 0x12;
pub const VX_KERNEL_GAUSSIAN_5x5: i32 = 0x13;
pub const VX_KERNEL_HARRIS_CORNERS: i32 = 0x14;
pub const VX_KERNEL_FAST_CORNERS: i32 = 0x15;
pub const VX_KERNEL_OPTICAL_FLOW_PYR_LK: i32 = 0x16;
pub const VX_KERNEL_LAPLACIAN: i32 = 0x17;
pub const VX_KERNEL_NON_LINEAR_FILTER: i32 = 0x18;
pub const VX_KERNEL_DILATE_3x3: i32 = 0x19;
pub const VX_KERNEL_ERODE_3x3: i32 = 0x1A;
pub const VX_KERNEL_HISTOGRAM: i32 = 0x1C;
pub const VX_KERNEL_EQUALIZE_HISTOGRAM: i32 = 0x1D;
pub const VX_KERNEL_INTEGRAL_IMAGE: i32 = 0x1E;
pub const VX_KERNEL_MEAN_STDDEV: i32 = 0x1F;
pub const VX_KERNEL_MINMAXLOC: i32 = 0x20;
pub const VX_KERNEL_ABSDIFF: i32 = 0x21;
pub const VX_KERNEL_MEAN_SHIFT: i32 = 0x22;
pub const VX_KERNEL_THRESHOLD: i32 = 0x23;
pub const VX_KERNEL_INTEGRAL_IMAGE_SQ: i32 = 0x24;
pub const VX_KERNEL_DILATE_5x5: i32 = 0x25;
pub const VX_KERNEL_ERODE_5x5: i32 = 0x26;
pub const VX_KERNEL_GAUSSIAN_PYRAMID: i32 = 0x27;
pub const VX_KERNEL_LAPLACIAN_PYRAMID: i32 = 0x28;
pub const VX_KERNEL_LAPLACIAN_RECONSTRUCT: i32 = 0x29;
pub const VX_KERNEL_REMAP: i32 = 0x2A;
pub const VX_KERNEL_CORNER_MIN_EIGEN_VAL: i32 = 0x2B;
pub const VX_KERNEL_HOUGH_LINES_P: i32 = 0x2C;
pub const VX_KERNEL_CANNY_EDGE_DETECTOR: i32 = 0x2D;

/// Kernel name to enum mapping for vision kernels
pub const VISION_KERNELS: &[(&str, i32, u32)] = &[
    // Color conversions
    ("org.khronos.openvx.color_convert", VX_KERNEL_COLOR_CONVERT, 2),
    ("org.khronos.openvx.channel_extract", VX_KERNEL_CHANNEL_EXTRACT, 3),
    ("org.khronos.openvx.channel_combine", VX_KERNEL_CHANNEL_COMBINE, 4),
    // Gradient operations
    ("org.khronos.openvx.sobel_3x3", VX_KERNEL_SOBEL_3x3, 3),
    ("org.khronos.openvx.magnitude", VX_KERNEL_MAGNITUDE, 3),
    ("org.khronos.openvx.phase", VX_KERNEL_PHASE, 3),
    // Geometric
    ("org.khronos.openvx.scale_image", VX_KERNEL_SCALE_IMAGE, 3),
    ("org.khronos.openvx.warp_affine", VX_KERNEL_WARP_AFFINE, 4),
    ("org.khronos.openvx.warp_perspective", VX_KERNEL_WARP_PERSPECTIVE, 4),
    // Arithmetic
    ("org.khronos.openvx.add", VX_KERNEL_ADD, 4),
    ("org.khronos.openvx.subtract", VX_KERNEL_SUBTRACT, 4),
    ("org.khronos.openvx.multiply", VX_KERNEL_MULTIPLY, 4),
    ("org.khronos.openvx.weighted_average", VX_KERNEL_WEIGHTED_AVERAGE, 4),
    // Filters
    ("org.khronos.openvx.convolve", VX_KERNEL_CONVOLVE, 3),
    ("org.khronos.openvx.gaussian_3x3", VX_KERNEL_GAUSSIAN_3x3, 2),
    ("org.khronos.openvx.median_3x3", VX_KERNEL_MEDIAN_3x3, 2),
    // Extended filters
    ("org.khronos.openvx.sobel_5x5", VX_KERNEL_SOBEL_5x5, 3),
    ("org.khronos.openvx.box_3x3", VX_KERNEL_BOX_3x3, 2),
    ("org.khronos.openvx.gaussian_5x5", VX_KERNEL_GAUSSIAN_5x5, 2),
    // Feature detection
    ("org.khronos.openvx.harris_corners", VX_KERNEL_HARRIS_CORNERS, 4),
    ("org.khronos.openvx.fast_corners", VX_KERNEL_FAST_CORNERS, 3),
    ("org.khronos.openvx.optical_flow_pyr_lk", VX_KERNEL_OPTICAL_FLOW_PYR_LK, 7),
    ("org.khronos.openvx.laplacian", VX_KERNEL_LAPLACIAN, 3),
    ("org.khronos.openvx.non_linear_filter", VX_KERNEL_NON_LINEAR_FILTER, 4),
    // Morphology
    ("org.khronos.openvx.dilate_3x3", VX_KERNEL_DILATE_3x3, 2),
    ("org.khronos.openvx.erode_3x3", VX_KERNEL_ERODE_3x3, 2),
    // Statistics
    ("org.khronos.openvx.histogram", VX_KERNEL_HISTOGRAM, 2),
    ("org.khronos.openvx.equalize_histogram", VX_KERNEL_EQUALIZE_HISTOGRAM, 2),
    ("org.khronos.openvx.integral_image", VX_KERNEL_INTEGRAL_IMAGE, 2),
    ("org.khronos.openvx.mean_stddev", VX_KERNEL_MEAN_STDDEV, 4),
    ("org.khronos.openvx.minmaxloc", VX_KERNEL_MINMAXLOC, 6),
    // Additional operations
    ("org.khronos.openvx.absdiff", VX_KERNEL_ABSDIFF, 3),
    ("org.khronos.openvx.mean_shift", VX_KERNEL_MEAN_SHIFT, 5),
    ("org.khronos.openvx.threshold", VX_KERNEL_THRESHOLD, 3),
    ("org.khronos.openvx.integral_image_sq", VX_KERNEL_INTEGRAL_IMAGE_SQ, 2),
    ("org.khronos.openvx.dilate_5x5", VX_KERNEL_DILATE_5x5, 2),
    ("org.khronos.openvx.erode_5x5", VX_KERNEL_ERODE_5x5, 2),
    // Pyramids
    ("org.khronos.openvx.gaussian_pyramid", VX_KERNEL_GAUSSIAN_PYRAMID, 2),
    ("org.khronos.openvx.laplacian_pyramid", VX_KERNEL_LAPLACIAN_PYRAMID, 2),
    ("org.khronos.openvx.laplacian_reconstruct", VX_KERNEL_LAPLACIAN_RECONSTRUCT, 3),
    // Geometric
    ("org.khronos.openvx.remap", VX_KERNEL_REMAP, 4),
    // Extended feature detection
    ("org.khronos.openvx.corner_min_eigen_val", VX_KERNEL_CORNER_MIN_EIGEN_VAL, 3),
    ("org.khronos.openvx.hough_lines_p", VX_KERNEL_HOUGH_LINES_P, 6),
    // Object detection
    ("org.khronos.openvx.canny_edge_detector", VX_KERNEL_CANNY_EDGE_DETECTOR, 4),
];
