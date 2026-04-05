//! OpenVX Vision Kernel Enum Constants
//!
//! These are the kernel enumeration values from the OpenVX 1.3 specification.
//! Per OpenVX spec: VX_KERNEL_<name> = VX_KERNEL_BASE(VX_ID_KHRONOS, VX_LIBRARY_KHR_BASE) + offset
//! Since VX_ID_KHRONOS=0x000 and VX_LIBRARY_KHR_BASE=0x0, the base is 0x00000000
//! Kernel enums start at 0x1 (not 0x0).

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
/// Per OpenVX 1.3 spec (from vx_kernels.h)
pub const VX_KERNEL_COLOR_CONVERT: i32 = 0x01;
pub const VX_KERNEL_CHANNEL_EXTRACT: i32 = 0x02;
pub const VX_KERNEL_CHANNEL_COMBINE: i32 = 0x03;
pub const VX_KERNEL_SOBEL_3x3: i32 = 0x04;
pub const VX_KERNEL_MAGNITUDE: i32 = 0x05;
pub const VX_KERNEL_PHASE: i32 = 0x06;
pub const VX_KERNEL_SCALE_IMAGE: i32 = 0x07;
pub const VX_KERNEL_TABLE_LOOKUP: i32 = 0x08;
pub const VX_KERNEL_HISTOGRAM: i32 = 0x09;
pub const VX_KERNEL_EQUALIZE_HISTOGRAM: i32 = 0x0A;
pub const VX_KERNEL_ABSDIFF: i32 = 0x0B;
pub const VX_KERNEL_MEAN_STDDEV: i32 = 0x0C;
pub const VX_KERNEL_THRESHOLD: i32 = 0x0D;
pub const VX_KERNEL_INTEGRAL_IMAGE: i32 = 0x0E;
pub const VX_KERNEL_DILATE_3x3: i32 = 0x0F;
pub const VX_KERNEL_ERODE_3x3: i32 = 0x10;
pub const VX_KERNEL_MEDIAN_3x3: i32 = 0x11;
pub const VX_KERNEL_BOX_3x3: i32 = 0x12;
pub const VX_KERNEL_GAUSSIAN_3x3: i32 = 0x13;
pub const VX_KERNEL_CUSTOM_CONVOLUTION: i32 = 0x14;
pub const VX_KERNEL_GAUSSIAN_PYRAMID: i32 = 0x15;
// 0x16 - 0x18 not assigned in base spec
pub const VX_KERNEL_MINMAXLOC: i32 = 0x19;
pub const VX_KERNEL_CONVERTDEPTH: i32 = 0x1A;
pub const VX_KERNEL_CANNY_EDGE_DETECTOR: i32 = 0x1B;
pub const VX_KERNEL_AND: i32 = 0x1C;
pub const VX_KERNEL_OR: i32 = 0x1D;
pub const VX_KERNEL_XOR: i32 = 0x1E;
pub const VX_KERNEL_NOT: i32 = 0x1F;
pub const VX_KERNEL_MULTIPLY: i32 = 0x20;
pub const VX_KERNEL_ADD: i32 = 0x21;
pub const VX_KERNEL_SUBTRACT: i32 = 0x22;
pub const VX_KERNEL_WARP_AFFINE: i32 = 0x23;
pub const VX_KERNEL_WARP_PERSPECTIVE: i32 = 0x24;
pub const VX_KERNEL_HARRIS_CORNERS: i32 = 0x25;
pub const VX_KERNEL_FAST_CORNERS: i32 = 0x26;
pub const VX_KERNEL_OPTICAL_FLOW_PYR_LK: i32 = 0x27;
pub const VX_KERNEL_REMAP: i32 = 0x28;
pub const VX_KERNEL_HALFSCALE_GAUSSIAN: i32 = 0x29;

// OpenVX 1.1 additions
pub const VX_KERNEL_LAPLACIAN_PYRAMID: i32 = 0x2A;
pub const VX_KERNEL_LAPLACIAN_RECONSTRUCT: i32 = 0x2B;
pub const VX_KERNEL_NON_LINEAR_FILTER: i32 = 0x2C;

// OpenVX 1.0.2 addition
pub const VX_KERNEL_WEIGHTED_AVERAGE: i32 = 0x40;

// OpenVX 1.2 additions
pub const VX_KERNEL_WARP_BILINEAR: i32 = 0x2D;
pub const VX_KERNEL_MATCH_TEMPLATE: i32 = 0x2E;
pub const VX_KERNEL_LBP: i32 = 0x2F;
pub const VX_KERNEL_HOUGH_LINES: i32 = 0x30;
pub const VX_KERNEL_HOUGH_CIRCLES: i32 = 0x31;
pub const VX_KERNEL_BILATERAL_FILTER: i32 = 0x32;
pub const VX_KERNEL_NON_MAX_SUPPRESSION: i32 = 0x33;
pub const VX_KERNEL_TENSOR_TRANSPOSE: i32 = 0x34;
pub const VX_KERNEL_TENSOR_CONVERT_DEPTH: i32 = 0x35;
pub const VX_KERNEL_TENSOR_RESIZE: i32 = 0x36;
pub const VX_KERNEL_TENSOR_PADDING: i32 = 0x37;

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
    ("org.khronos.openvx.remap", VX_KERNEL_REMAP, 4),
    ("org.khronos.openvx.halfscale_gaussian", VX_KERNEL_HALFSCALE_GAUSSIAN, 3),
    // Filters
    ("org.khronos.openvx.gaussian_3x3", VX_KERNEL_GAUSSIAN_3x3, 2),
    ("org.khronos.openvx.box_3x3", VX_KERNEL_BOX_3x3, 2),
    ("org.khronos.openvx.median_3x3", VX_KERNEL_MEDIAN_3x3, 2),
    ("org.khronos.openvx.custom_convolution", VX_KERNEL_CUSTOM_CONVOLUTION, 3),
    ("org.khronos.openvx.gaussian_pyramid", VX_KERNEL_GAUSSIAN_PYRAMID, 2),
    // Morphology
    ("org.khronos.openvx.dilate_3x3", VX_KERNEL_DILATE_3x3, 2),
    ("org.khronos.openvx.erode_3x3", VX_KERNEL_ERODE_3x3, 2),
    // Arithmetic
    ("org.khronos.openvx.add", VX_KERNEL_ADD, 4),
    ("org.khronos.openvx.subtract", VX_KERNEL_SUBTRACT, 4),
    ("org.khronos.openvx.multiply", VX_KERNEL_MULTIPLY, 7),
    // Bitwise
    ("org.khronos.openvx.and", VX_KERNEL_AND, 3),
    ("org.khronos.openvx.or", VX_KERNEL_OR, 3),
    ("org.khronos.openvx.xor", VX_KERNEL_XOR, 3),
    ("org.khronos.openvx.not", VX_KERNEL_NOT, 2),
    // Statistics
    ("org.khronos.openvx.histogram", VX_KERNEL_HISTOGRAM, 2),
    ("org.khronos.openvx.equalize_histogram", VX_KERNEL_EQUALIZE_HISTOGRAM, 2),
    ("org.khronos.openvx.integral_image", VX_KERNEL_INTEGRAL_IMAGE, 2),
    ("org.khronos.openvx.mean_stddev", VX_KERNEL_MEAN_STDDEV, 4),
    ("org.khronos.openvx.minmaxloc", VX_KERNEL_MINMAXLOC, 6),
    // Additional operations
    ("org.khronos.openvx.absdiff", VX_KERNEL_ABSDIFF, 3),
    ("org.khronos.openvx.threshold", VX_KERNEL_THRESHOLD, 3),
    ("org.khronos.openvx.table_lookup", VX_KERNEL_TABLE_LOOKUP, 3),
    ("org.khronos.openvx.convertdepth", VX_KERNEL_CONVERTDEPTH, 4),
    // Feature detection
    ("org.khronos.openvx.harris_corners", VX_KERNEL_HARRIS_CORNERS, 7),
    ("org.khronos.openvx.fast_corners", VX_KERNEL_FAST_CORNERS, 5),
    ("org.khronos.openvx.optical_flow_pyr_lk", VX_KERNEL_OPTICAL_FLOW_PYR_LK, 7),
    ("org.khronos.openvx.canny_edge_detector", VX_KERNEL_CANNY_EDGE_DETECTOR, 5),
    // OpenVX 1.1
    ("org.khronos.openvx.laplacian_pyramid", VX_KERNEL_LAPLACIAN_PYRAMID, 2),
    ("org.khronos.openvx.laplacian_reconstruct", VX_KERNEL_LAPLACIAN_RECONSTRUCT, 3),
    ("org.khronos.openvx.non_linear_filter", VX_KERNEL_NON_LINEAR_FILTER, 4),
    // OpenVX 1.0.2
    ("org.khronos.openvx.weighted_average", VX_KERNEL_WEIGHTED_AVERAGE, 4),
];
