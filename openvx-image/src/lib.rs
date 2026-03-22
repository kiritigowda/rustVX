//! Image module

pub mod c_api;

use openvx_core::{VxResult, Referenceable, VxType};
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

/// Image format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Rgb,
    Rgba,
    Gray,
    NV12,
    NV21,
}

impl ImageFormat {
    pub fn channels(&self) -> usize {
        match self {
            ImageFormat::Gray => 1,
            ImageFormat::Rgb => 3,
            ImageFormat::Rgba => 4,
            ImageFormat::NV12 | ImageFormat::NV21 => 3,
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
        let channels = format.channels();
        let data = vec![0u8; width * height * channels];
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
    let data = vec![value; width * height * format.channels()];
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
