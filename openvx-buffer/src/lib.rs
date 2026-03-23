//! Buffer module

pub mod c_api;

use openvx_core::{VxResult, Referenceable, VxType};

/// Generic buffer for intermediate data
pub struct Buffer {
    data: Vec<u8>,
    width: usize,
    height: usize,
    stride: usize,
}

impl Buffer {
    pub fn new(width: usize, height: usize) -> Self {
        let stride = width;
        // Use saturating_mul to prevent integer overflow
        let size = width.saturating_mul(height);
        Buffer {
            data: vec![0u8; size],
            width,
            height,
            stride,
        }
    }
    
    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn stride(&self) -> usize { self.stride }
    pub fn data(&self) -> &[u8] { &self.data }
    pub fn data_mut(&mut self) -> &mut [u8] { &mut self.data }
    
    pub fn get(&self, x: usize, y: usize) -> u8 {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x]
        } else {
            0
        }
    }
    
    pub fn set(&mut self, x: usize, y: usize, value: u8) {
        if x < self.width && y < self.height {
            self.data[y * self.stride + x] = value;
        }
    }
    
    pub fn get_i16(&self, x: usize, y: usize) -> i16 {
        let idx = (y * self.stride + x) * 2;
        if idx + 1 < self.data.len() {
            i16::from_le_bytes([self.data[idx], self.data[idx + 1]])
        } else {
            0
        }
    }
    
    pub fn set_i16(&mut self, x: usize, y: usize, value: i16) {
        let idx = (y * self.stride + x) * 2;
        if idx + 1 < self.data.len() {
            let bytes = value.to_le_bytes();
            self.data[idx] = bytes[0];
            self.data[idx + 1] = bytes[1];
        }
    }
    
    pub fn get_f32(&self, x: usize, y: usize) -> f32 {
        let idx = (y * self.stride + x) * 4;
        if idx + 3 < self.data.len() {
            let bytes = [
                self.data[idx], self.data[idx + 1],
                self.data[idx + 2], self.data[idx + 3]
            ];
            f32::from_le_bytes(bytes)
        } else {
            0.0
        }
    }
    
    pub fn set_f32(&mut self, x: usize, y: usize, value: f32) {
        let idx = (y * self.stride + x) * 4;
        if idx + 3 < self.data.len() {
            let bytes = value.to_le_bytes();
            self.data[idx] = bytes[0];
            self.data[idx + 1] = bytes[1];
            self.data[idx + 2] = bytes[2];
            self.data[idx + 3] = bytes[3];
        }
    }
}

impl Referenceable for Buffer {
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn get_type(&self) -> VxType { VxType::Buffer }
    fn get_reference_count(&self) -> usize { 1 }
    fn retain(&self) {}
    fn release(&self) -> usize { 1 }
    fn get_context_id(&self) -> u32 { 1 }
    fn get_id(&self) -> u64 { 1 }
    fn query_attribute(&self, _attr: u32, _val: &mut [u8]) -> VxResult<()> { Ok(()) }
}
