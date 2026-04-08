//! Image module

pub mod c_api;
pub mod ct_image;

// Re-export from c_api
pub use c_api::vxCreateImageFromChannel;
pub use c_api::vxCloneImage;
pub use c_api::vxCloneImageWithGraph;

// Re-export CT Image functions for conformance testing
pub use ct_image::{
    CT_Rect, CT_Image, CT_ImageCopyDirection, CT_ImageData,
    ct_channels, ct_stride_bytes, ct_image_bits_per_pixel,
    ct_get_num_planes, ct_image_get_plane_base,
    ct_image_get_channel_step_x, ct_image_get_channel_step_y,
    ct_image_get_channel_subsampling_x, ct_image_get_channel_subsampling_y,
    ct_image_get_channel_plane, ct_image_get_channel_component,
    ct_get_num_channels, ct_allocate_image, ct_allocate_image_hdr,
    ct_get_image_roi, ct_get_image_roi_, ct_adjust_roi,
    ct_image_data_ptr_1u, ct_image_data_replicate_1u, ct_image_data_constant_1u,
    ct_image_data_ptr_8u, ct_image_data_replicate_8u, ct_image_data_constant_8u,
    ct_free_image, ct_image_to_vx_image, ct_image_from_vx_image, ct_image_copy,
    ct_image_create_clone, ct_fill_ct_image_random, ct_allocate_ct_image_random,
    U8_ct_image_to_U1_ct_image, U1_ct_image_to_U8_ct_image,
    threshold_U8_ct_image, ct_assert_eq_ctimage, ct_dump_image_info,
    ct_read_image, ct_write_image, ct_image_read_rect_S32,
};

use openvx_core::{VxResult, Referenceable, VxType};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Image format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Rgb,
    Rgba,
    Gray,
    S16, // Signed 16-bit for gradients
    NV12,
    NV21,
    IYUV,
    YUV4,
}

impl ImageFormat {
    pub fn channels(&self) -> usize {
        match self {
            ImageFormat::Gray | ImageFormat::S16 => 1,
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
            // NV12, NV21, IYUV are planar formats with different sizing
            // These need special handling - using 1 here as default for buffer calculation
            // Planar formats should use format.buffer_size(width, height) instead
            ImageFormat::NV12 | ImageFormat::NV21 | ImageFormat::IYUV | ImageFormat::YUV4 => 1,
        }
    }
    
    /// Calculate buffer size for this format with given dimensions
    /// For planar formats, this accounts for subsampled chroma planes
    pub fn buffer_size(&self, width: usize, height: usize) -> usize {
        match self {
            ImageFormat::Gray => width.saturating_mul(height),
            ImageFormat::S16 => width.saturating_mul(height).saturating_mul(2), // 2 bytes per pixel
            ImageFormat::Rgb => width.saturating_mul(height).saturating_mul(3),
            ImageFormat::Rgba => width.saturating_mul(height).saturating_mul(4),
            // IYUV/I420: Y (full) + U (quarter) + V (quarter) = 1.5 * width * height
            ImageFormat::IYUV => {
                let y_size = width.saturating_mul(height);
                let half_w = (width + 1) / 2;
                let half_h = (height + 1) / 2;
                let uv_size = half_w.saturating_mul(half_h);
                y_size.saturating_add(uv_size).saturating_add(uv_size)
            }
            // NV12/NV21: Y (full) + UV interleaved (quarter * 2) = 1.5 * width * height
            ImageFormat::NV12 | ImageFormat::NV21 => {
                let y_size = width.saturating_mul(height);
                let half_h = (height + 1) / 2;
                let uv_stride = width; // NV12 uses full width stride for UV
                let uv_size = uv_stride.saturating_mul(half_h);
                y_size.saturating_add(uv_size)
            }
            // YUV4: Three full-size planes = 3 * width * height
            ImageFormat::YUV4 => width.saturating_mul(height).saturating_mul(3),
        }
    }
}

/// Image struct
pub struct Image {
    width: usize,
    height: usize,
    format: ImageFormat,
    data: RwLock<Vec<u8>>,
}

impl Image {
    pub fn new(width: usize, height: usize, format: ImageFormat) -> Self {
        // Use format-specific buffer size calculation for planar formats
        let size = format.buffer_size(width, height);
        
        // Add sanity limit to prevent massive allocations
        const MAX_ALLOCATION_SIZE: usize = 1024 * 1024 * 1024; // 1GB
        if size > MAX_ALLOCATION_SIZE {
            panic!("Image allocation size {}x{} format {:?} = {} exceeds maximum allocation limit", width, height, format, size);
        }
        if size == 0 {
            panic!("Image allocation size is 0 for {}x{} format {:?}", width, height, format);
        }
        
        let data = vec![0u8; size];
        Image {
            width,
            height,
            format,
            data: RwLock::new(data),
        }
    }
    
    pub fn from_data(width: usize, height: usize, format: ImageFormat, data: Vec<u8>) -> Self {
        Image {
            width,
            height,
            format,
            data: RwLock::new(data),
        }
    }
    
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn format(&self) -> ImageFormat { self.format }
    
    pub fn data(&self) -> RwLockReadGuard<'_, Vec<u8>> {
        self.data.read().unwrap()
    }
    
    pub fn data_mut(&self) -> RwLockWriteGuard<'_, Vec<u8>> {
        self.data.write().unwrap()
    }
    
    pub fn get_pixel(&self, x: usize, y: usize) -> u8 {
        self.data.read().unwrap()[y * self.width + x]
    }
    
    pub fn set_pixel(&self, x: usize, y: usize, value: u8) {
        self.data.write().unwrap()[y * self.width + x] = value;
    }
    
    pub fn get_rgb(&self, x: usize, y: usize) -> (u8, u8, u8) {
        let idx = (y * self.width + x) * 3;
        let data = self.data.read().unwrap();
        (data[idx], data[idx + 1], data[idx + 2])
    }
    
    pub fn set_rgb(&self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        let idx = (y * self.width + x) * 3;
        let mut data = self.data.write().unwrap();
        data[idx] = r;
        data[idx + 1] = g;
        data[idx + 2] = b;
    }
    
    pub fn pixels(&self) -> Vec<u8> {
        self.data.read().unwrap().iter().cloned().collect()
    }
    
    /// Get pixel as i16 (for S16 format)
    pub fn get_pixel_i16(&self, x: usize, y: usize) -> i16 {
        let idx = (y * self.width + x) * 2; // 2 bytes per i16
        let data = self.data.read().unwrap();
        i16::from_le_bytes([data[idx], data[idx + 1]])
    }
    
    /// Set pixel as i16 (for S16 format)
    pub fn set_pixel_i16(&self, x: usize, y: usize, value: i16) {
        let idx = (y * self.width + x) * 2; // 2 bytes per i16
        let bytes = value.to_le_bytes();
        let mut data = self.data.write().unwrap();
        data[idx] = bytes[0];
        data[idx + 1] = bytes[1];
    }
    
    /// Get mutable data as i16 slice (for S16 format)
    pub fn data_mut_i16(&self) -> Vec<i16> {
        let data = self.data.write().unwrap();
        data.chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
    }
}

impl Referenceable for Image {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn get_type(&self) -> VxType { VxType::Image }
    fn get_reference_count(&self) -> usize { 1 }
    fn retain(&self) {}
    fn release(&self) -> usize { 1 }
    fn get_context_id(&self) -> u32 { 1 }
    fn get_id(&self) -> u64 { 1 }
    fn query_attribute(&self, _attr: u32, _val: &mut [u8]) -> VxResult<()> { Ok(()) }
}

/// Create a uniform image
pub fn create_uniform_image(width: usize, height: usize, format: ImageFormat, value: u8) -> Image {
    // Use format-specific buffer size calculation for planar formats
    let size = format.buffer_size(width, height);
    
    // Add sanity limit to prevent massive allocations
    const MAX_ALLOCATION_SIZE: usize = 1024 * 1024 * 1024; // 1GB
    if size > MAX_ALLOCATION_SIZE {
        panic!("Image allocation size {}x{} format {:?} = {} exceeds maximum allocation limit", width, height, format, size);
    }
    if size == 0 {
        panic!("Image allocation size is 0 for {}x{} format {:?}", width, height, format);
    }
    
    let data = vec![value; size];
    Image::from_data(width, height, format, data)
}

/// Create a test pattern image
pub fn create_test_image(width: usize, height: usize) -> Image {
    let img = Image::new(width, height, ImageFormat::Gray);
    for y in 0..height {
        for x in 0..width {
            let val = ((x * 255) / width) as u8;
            img.set_pixel(x, y, val);
        }
    }
    img
}

/// Clone an image - creates a deep copy
/// 
/// This creates a new image with the same dimensions and format,
/// then copies all pixel data from the source.
pub fn clone_image(source: &Image) -> Image {
    let source_data = source.data.read().unwrap();
    let cloned_data = source_data.clone();
    
    Image {
        width: source.width,
        height: source.height,
        format: source.format,
        data: RwLock::new(cloned_data),
    }
}
<<<<<<< HEAD
=======

/// Pyramid structure
/// A pyramid is a multi-level image representation where each level
/// is a scaled-down version of the previous level.
pub struct Pyramid {
    num_levels: usize,
    scale: f32,
    levels: Vec<std::sync::Arc<Image>>,
}

impl Pyramid {
    pub fn new(num_levels: usize, scale: f32, levels: Vec<std::sync::Arc<Image>>) -> Self {
        Pyramid {
            num_levels,
            scale,
            levels,
        }
    }
    
    pub fn num_levels(&self) -> usize { self.num_levels }
    pub fn scale(&self) -> f32 { self.scale }
    pub fn levels(&self) -> &Vec<std::sync::Arc<Image>> { &self.levels }
    
    pub fn get_level(&self, index: usize) -> Option<&std::sync::Arc<Image>> {
        self.levels.get(index)
    }
}
>>>>>>> origin/master
