//! Color conversion tests

use openvx_image::{Image, ImageFormat, create_test_image};
use openvx_vision::color::*;

#[test]
fn test_rgb_to_gray() {
    let rgb = Image::new(4, 4, ImageFormat::Rgb);
    let gray = Image::new(4, 4, ImageFormat::Gray);
    
    let result = rgb_to_gray(&rgb, &gray);
    assert!(result.is_ok());
}

#[test]
fn test_gray_to_rgb() {
    let gray = Image::new(4, 4, ImageFormat::Gray);
    let rgb = Image::new(4, 4, ImageFormat::Rgb);
    
    let result = gray_to_rgb(&gray, &rgb);
    assert!(result.is_ok());
}

#[test]
fn test_rgb_to_nv12() {
    let rgb = Image::new(4, 4, ImageFormat::Rgb);
    let nv12 = Image::new(4, 4, ImageFormat::NV12);
    
    let result = rgb_to_nv12(&rgb, &nv12);
    assert!(result.is_ok());
}

#[test]
fn test_nv12_to_rgb() {
    let nv12 = Image::new(4, 4, ImageFormat::NV12);
    let rgb = Image::new(4, 4, ImageFormat::Rgb);
    
    let result = nv12_to_rgb(&nv12, &rgb);
    assert!(result.is_ok());
}

#[test]
fn test_color_conversion_consistency() {
    // Create test image with known values
    let rgb = Image::new(4, 4, ImageFormat::Rgb);
    for y in 0..4 {
        for x in 0..4 {
            let val = (y * 4 + x) as u8 * 16;
            rgb.set_rgb(x, y, val, val, val);
        }
    }
    
    let gray = Image::new(4, 4, ImageFormat::Gray);
    rgb_to_gray(&rgb, &gray).unwrap();
    
    // When R=G=B, gray should equal that value
    let gray_data = gray.data();
    for y in 0..4 {
        for x in 0..4 {
            let expected = (y * 4 + x) as u8 * 16;
            // The actual gray value should be close to the input (BT.709 weighted)
            let actual = gray_data[y * 4 + x];
            let diff = (expected as i16 - actual as i16).abs();
            assert!(diff < 5, "Color conversion inconsistency at ({}, {}): expected {}, got {}", x, y, expected, actual);
        }
    }
}
