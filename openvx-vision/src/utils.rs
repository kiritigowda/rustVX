//! Utility functions for vision algorithms

use openvx_image::Image;

/// Border modes for filter operations
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BorderMode {
    Undefined,          // Skip border pixels
    Constant(u8),       // Fill with constant value
    Replicate,          // Replicate edge pixels
}

impl Default for BorderMode {
    fn default() -> Self { BorderMode::Undefined }
}

/// Get pixel with border handling
pub fn get_pixel_bordered(img: &Image, x: isize, y: isize, border: BorderMode) -> u8 {
    let width = img.width() as isize;
    let height = img.height() as isize;
    
    if x >= 0 && x < width && y >= 0 && y < height {
        img.get_pixel(x as usize, y as usize)
    } else {
        match border {
            BorderMode::Undefined => 0,
            BorderMode::Constant(val) => val,
            BorderMode::Replicate => {
                let cx = x.max(0).min(width - 1) as usize;
                let cy = y.max(0).min(height - 1) as usize;
                img.get_pixel(cx, cy)
            }
        }
    }
}

/// Clamp value to range
#[inline]
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min { min } else if value > max { max } else { value }
}

/// Clamp to u8 range
#[inline]
pub fn clamp_u8(value: i32) -> u8 {
    clamp(value, 0, 255) as u8
}

/// Quickselect for median
pub fn quickselect(arr: &mut [u8], k: usize) -> u8 {
    if k >= arr.len() {
        return 0;
    }
    
    let len = arr.len();
    let mut left = 0;
    let mut right = len - 1;
    
    loop {
        if left == right {
            return arr[left];
        }
        
        let mut pivot_index = partition(arr, left, right);
        
        if k == pivot_index {
            return arr[k];
        } else if k < pivot_index {
            if pivot_index == 0 { break; }
            right = pivot_index - 1;
        } else {
            left = pivot_index + 1;
        }
    }
    
    arr[k]
}

fn partition(arr: &mut [u8], left: usize, right: usize) -> usize {
    let pivot_value = arr[right];
    let mut store_index = left;
    
    for i in left..right {
        if arr[i] < pivot_value {
            arr.swap(store_index, i);
            store_index += 1;
        }
    }
    
    arr.swap(store_index, right);
    store_index
}

/// Fixed point math (Q14 format)
pub const Q14: i32 = 14;
pub const Q14_SCALE: i32 = 1 << Q14;

/// Convert float to Q14
#[inline]
pub fn f32_to_q14(v: f32) -> i32 {
    (v * Q14_SCALE as f32) as i32
}

/// Convert Q14 to float
#[inline]
pub fn q14_to_f32(v: i32) -> f32 {
    v as f32 / Q14_SCALE as f32
}

/// Multiply two Q14 values
#[inline]
pub fn q14_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> Q14) as i32
}

/// Divide two Q14 values
#[inline]
pub fn q14_div(a: i32, b: i32) -> i32 {
    (((a as i64) << Q14) / b.max(1) as i64) as i32
}

/// Bilinear interpolation
pub fn bilinear_interpolate(img: &Image, x: f32, y: f32) -> u8 {
    let width = img.width() as i32;
    let height = img.height() as i32;
    
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    
    if x0 < 0 || y0 < 0 || x0 >= width || y0 >= height {
        return 0;
    }
    
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    
    let p00 = img.get_pixel(x0 as usize, y0 as usize) as f32;
    let p10 = img.get_pixel(x1 as usize, y0 as usize) as f32;
    let p01 = img.get_pixel(x0 as usize, y1 as usize) as f32;
    let p11 = img.get_pixel(x1 as usize, y1 as usize) as f32;
    
    let value = (1.0 - fx) * (1.0 - fy) * p00 +
                fx * (1.0 - fy) * p10 +
                (1.0 - fx) * fy * p01 +
                fx * fy * p11;
    
    clamp(value as i32, 0, 255) as u8
}

/// Nearest neighbor interpolation
pub fn nearest_interpolate(img: &Image, x: f32, y: f32) -> u8 {
    let x = clamp(x.round() as i32, 0, img.width() as i32 - 1) as usize;
    let y = clamp(y.round() as i32, 0, img.height() as i32 - 1) as usize;
    img.get_pixel(x, y)
}

/// Gaussian kernel (3x3)
pub const GAUSSIAN_3X3: [i32; 3] = [1, 2, 1];

/// Gaussian kernel (5x5)
pub const GAUSSIAN_5X5: [i32; 5] = [1, 4, 6, 4, 1];

/// Sobel X kernel (3x3)
pub const SOBEL_X: [[i32; 3]; 3] = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
];

/// Sobel Y kernel (3x3)
pub const SOBEL_Y: [[i32; 3]; 3] = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1],
];

/// Compute image integral
pub fn integral_image(src: &Image, dst: &mut Vec<u32>) {
    let width = src.width();
    let height = src.height();
    dst.resize(width * height, 0);
    
    for y in 0..height {
        let mut row_sum = 0u32;
        for x in 0..width {
            row_sum += src.get_pixel(x, y) as u32;
            if y == 0 {
                dst[y * width + x] = row_sum;
            } else {
                dst[y * width + x] = dst[(y - 1) * width + x] + row_sum;
            }
        }
    }
}
