//! Gradient operation tests

use openvx_image::{Image, ImageFormat};
use openvx_vision::gradient::*;

#[test]
fn test_sobel3x3_horizontal_edge() {
    // Create image with horizontal edge
    let input = Image::new(5, 5, ImageFormat::Gray);
    for y in 0..5 {
        for x in 0..5 {
            let val = if y < 2 { 50 } else { 200 };
            input.set_pixel(x, y, val);
        }
    }

    let grad_x = Image::new(5, 5, ImageFormat::Gray);
    let grad_y = Image::new(5, 5, ImageFormat::Gray);

    sobel3x3(&input, &grad_x, &grad_y).unwrap();

    // At the edge (y=2), grad_y should be strong
    let gy_data = grad_y.data();

    // Center of edge row should have strong gradient
    let edge_val = gy_data[2 * 5 + 2];
    assert!(edge_val > 128, "Sobel Y should detect horizontal edge");
}

#[test]
fn test_sobel3x3_vertical_edge() {
    // Create image with vertical edge
    let input = Image::new(5, 5, ImageFormat::Gray);
    for y in 0..5 {
        for x in 0..5 {
            let val = if x < 2 { 50 } else { 200 };
            input.set_pixel(x, y, val);
        }
    }

    let grad_x = Image::new(5, 5, ImageFormat::Gray);
    let grad_y = Image::new(5, 5, ImageFormat::Gray);

    sobel3x3(&input, &grad_x, &grad_y).unwrap();

    // At the edge (x=2), grad_x should be strong
    let gx_data = grad_x.data();

    // Center of edge column should have strong gradient
    let edge_val = gx_data[2 * 5 + 2];
    assert!(edge_val > 128, "Sobel X should detect vertical edge");
}

#[test]
fn test_magnitude_computation() {
    // Create synthetic gradients
    let width = 4;
    let height = 4;

    let grad_x = Image::new(width, height, ImageFormat::Gray);
    let grad_y = Image::new(width, height, ImageFormat::Gray);
    let mag = Image::new(width, height, ImageFormat::Gray);

    // Set up known gradient values
    for y in 0..height {
        for x in 0..width {
            // gradient x = 3, gradient y = 4
            // magnitude should be 5
            grad_x.set_pixel(x, y, 131); // 128 + 3
            grad_y.set_pixel(x, y, 132); // 128 + 4
        }
    }

    magnitude(&grad_x, &grad_y, &mag).unwrap();

    // Verify magnitude is computed
    let mag_data = mag.data();

    // sqrt(3^2 + 4^2) = 5
    let expected = 5i32;
    for y in 0..height {
        for x in 0..width {
            let actual = mag_data[y * width + x] as i32;
            let diff = (actual - expected).abs();
            assert!(
                diff <= 2,
                "Magnitude mismatch at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_phase_computation() {
    let width = 4;
    let height = 4;

    let grad_x = Image::new(width, height, ImageFormat::Gray);
    let grad_y = Image::new(width, height, ImageFormat::Gray);
    let phase = Image::new(width, height, ImageFormat::Gray);

    // Diagonal gradient (45 degrees): grad_x = grad_y
    for y in 0..height {
        for x in 0..width {
            grad_x.set_pixel(x, y, 138); // 128 + 10
            grad_y.set_pixel(x, y, 138); // 128 + 10
        }
    }

    phase_op(&grad_x, &grad_y, &phase).unwrap();

    // Phase should be 45 degrees
    // 45 degrees = 255 * (45/360) ≈ 32
    let phase_data = phase.data();

    let expected = (255.0 * 45.0 / 360.0) as u8;
    for &v in phase_data.iter() {
        let diff = (v as i16 - expected as i16).abs();
        assert!(
            diff <= 10,
            "Phase computation mismatch: expected around {}, got {}",
            expected,
            v
        );
    }
}

#[test]
fn test_gradient_symmetry() {
    // Uniform image should have zero gradients
    let input = Image::new(5, 5, ImageFormat::Gray);
    // All zeros - uniform

    let grad_x = Image::new(5, 5, ImageFormat::Gray);
    let grad_y = Image::new(5, 5, ImageFormat::Gray);

    sobel3x3(&input, &grad_x, &grad_y).unwrap();

    let gx_data = grad_x.data();
    let gy_data = grad_y.data();

    // Interior should have 128 (offset zero)
    for y in 1..4 {
        for x in 1..4 {
            let gx = gx_data[y * 5 + x];
            let gy = gy_data[y * 5 + x];
            assert!(
                gx.abs_diff(128) <= 2,
                "Uniform image should have zero gradient X"
            );
            assert!(
                gy.abs_diff(128) <= 2,
                "Uniform image should have zero gradient Y"
            );
        }
    }
}
