//! Morphological operation tests

use openvx_image::{Image, ImageFormat};
use openvx_vision::morphology::*;
use openvx_vision::utils::BorderMode;

#[test]
fn test_dilate_expands_white() {
    let input = Image::new(5, 5, ImageFormat::Gray);
    let output = Image::new(5, 5, ImageFormat::Gray);

    // Single white pixel
    input.set_pixel(2, 2, 255);

    dilate3x3(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Dilation should expand the white pixel to 3x3
    for y in 1..4 {
        for x in 1..4 {
            assert_eq!(
                output_data[y * 5 + x],
                255,
                "Dilation should expand white pixel at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_erode_shrinks_white() {
    let input = Image::new(5, 5, ImageFormat::Gray);
    let output = Image::new(5, 5, ImageFormat::Gray);

    // Fill with white except edges
    for y in 0..5 {
        for x in 0..5 {
            input.set_pixel(x, y, 255);
        }
    }

    // Single black pixel in center
    input.set_pixel(2, 2, 0);

    erode3x3(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Erosion should shrink white regions - the black pixel should expand
    // Check center is now black
    assert_eq!(
        output_data[2 * 5 + 2],
        0,
        "Erosion should expand black pixel"
    );
}

#[test]
fn test_dilate_on_uniform_white() {
    let input = Image::new(5, 5, ImageFormat::Gray);
    // All zeros (black)
    let output = Image::new(5, 5, ImageFormat::Gray);

    dilate3x3(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Dilation of black should stay black
    for &v in output_data.iter() {
        assert_eq!(v, 0, "Dilation of black should stay black");
    }
}

#[test]
fn test_erode_on_uniform_white() {
    let input = Image::new(5, 5, ImageFormat::Gray);
    // All 255 (white)
    for i in 0..25 {
        input.set_pixel(i % 5, i / 5, 255);
    }

    let output = Image::new(5, 5, ImageFormat::Gray);
    erode3x3(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Erosion of white should stay white
    for &v in output_data.iter() {
        assert_eq!(v, 255, "Erosion of white should stay white");
    }
}

#[test]
fn test_opening_removes_small_noise() {
    let input = Image::new(7, 7, ImageFormat::Gray);

    // Black background with single white pixel
    input.set_pixel(3, 3, 255);

    let output = Image::new(7, 7, ImageFormat::Gray);
    opening3x3(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Opening should remove small white spots
    assert_eq!(
        output_data[3 * 7 + 3],
        0,
        "Opening should remove isolated white pixel"
    );
}

#[test]
fn test_closing_fills_small_holes() {
    let input = Image::new(7, 7, ImageFormat::Gray);

    // White background with single black pixel
    for y in 0..7 {
        for x in 0..7 {
            input.set_pixel(x, y, 255);
        }
    }
    input.set_pixel(3, 3, 0);

    let output = Image::new(7, 7, ImageFormat::Gray);
    closing3x3(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Closing should fill small black holes
    assert_eq!(
        output_data[3 * 7 + 3],
        255,
        "Closing should fill isolated black pixel"
    );
}

#[test]
fn test_morphological_gradient() {
    let input = Image::new(5, 5, ImageFormat::Gray);

    // Gradient should be high at edges
    for y in 0..5 {
        for x in 0..5 {
            let val = if x >= 2 { 200 } else { 50 };
            input.set_pixel(x, y, val);
        }
    }

    let output = Image::new(5, 5, ImageFormat::Gray);
    morphological_gradient(&input, &output, BorderMode::Replicate).unwrap();

    let output_data = output.data();

    // Gradient should be non-zero at edge
    for y in 0..5 {
        let edge_val = output_data[y * 5 + 2];
        assert!(
            edge_val > 0,
            "Gradient should be non-zero at edge (y={}, x=2)",
            y
        );
    }
}
