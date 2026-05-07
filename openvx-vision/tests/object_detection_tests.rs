//! Tests for object detection operations

use openvx_image::{Image, ImageFormat};
use openvx_vision::object_detection::{canny_edge_detector, threshold, LineSegment};

#[test]
fn test_threshold_basic() {
    let input = Image::new(5, 5, ImageFormat::Gray);
    let output = Image::new(5, 5, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = if i < 12 { 50 } else { 200 };
        }
    }

    threshold(&input, &output, 100, 255).unwrap();

    let out_data = output.data();
    // First half should be 0 (below threshold), second half should be 255
    for i in 0..12 {
        assert_eq!(out_data[i], 0, "Pixel {} should be 0", i);
    }
    for i in 12..25 {
        assert_eq!(out_data[i], 255, "Pixel {} should be 255", i);
    }
}

#[test]
fn test_threshold_all_below() {
    let input = Image::new(3, 3, ImageFormat::Gray);
    let output = Image::new(3, 3, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = 50;
        }
    }

    threshold(&input, &output, 100, 255).unwrap();

    let out_data = output.data();
    for i in 0..out_data.len() {
        assert_eq!(out_data[i], 0);
    }
}

#[test]
fn test_threshold_all_above() {
    let input = Image::new(3, 3, ImageFormat::Gray);
    let output = Image::new(3, 3, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = 150;
        }
    }

    threshold(&input, &output, 100, 255).unwrap();

    let out_data = output.data();
    for i in 0..out_data.len() {
        assert_eq!(out_data[i], 255);
    }
}

#[test]
fn test_canny_edge_detector() {
    let input = Image::new(10, 10, ImageFormat::Gray);
    let output = Image::new(10, 10, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        // Create a simple edge: left half black, right half white
        for y in 0..10 {
            for x in 0..10 {
                data[y * 10 + x] = if x < 5 { 0 } else { 255 };
            }
        }
    }

    canny_edge_detector(&input, &output, 50, 150).unwrap();

    let out_data = output.data();
    // The output should contain edges (non-zero values near the edge)
    let edge_pixels: Vec<&u8> = out_data.iter().filter(|v| **v > 0).collect();
    assert!(!edge_pixels.is_empty(), "Canny should detect edges");
}

#[test]
fn test_canny_on_uniform_image() {
    let input = Image::new(10, 10, ImageFormat::Gray);
    let output = Image::new(10, 10, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = 128;
        }
    }

    canny_edge_detector(&input, &output, 50, 150).unwrap();

    let out_data = output.data();
    // Uniform image should have no edges
    let edge_pixels: Vec<&u8> = out_data.iter().filter(|v| **v > 0).collect();
    assert!(edge_pixels.is_empty(), "Uniform image should have no edges");
}

#[test]
fn test_line_segment_creation() {
    let line = LineSegment::new(0, 0, 10, 10);

    assert_eq!(line.x1, 0);
    assert_eq!(line.y1, 0);
    assert_eq!(line.x2, 10);
    assert_eq!(line.y2, 10);
}

#[test]
fn test_line_segment_length() {
    let line1 = LineSegment::new(0, 0, 3, 4);
    assert!((line1.length() - 5.0).abs() < 0.01);

    let line2 = LineSegment::new(0, 0, 0, 10);
    assert!((line2.length() - 10.0).abs() < 0.01);

    let line3 = LineSegment::new(0, 0, 10, 0);
    assert!((line3.length() - 10.0).abs() < 0.01);
}

#[test]
fn test_line_segment_angle() {
    let horizontal = LineSegment::new(0, 0, 10, 0);
    assert!((horizontal.angle()).abs() < 0.01);

    let vertical = LineSegment::new(0, 0, 0, 10);
    assert!((vertical.angle() - std::f32::consts::PI / 2.0).abs() < 0.01);

    let diagonal = LineSegment::new(0, 0, 10, 10);
    assert!((diagonal.angle() - std::f32::consts::PI / 4.0).abs() < 0.01);
}
