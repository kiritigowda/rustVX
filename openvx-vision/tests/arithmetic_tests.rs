//! Arithmetic operation tests

use openvx_image::{Image, ImageFormat};
use openvx_vision::arithmetic::*;

#[test]
fn test_add_basic() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with test values
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 100);
            src2.set_pixel(x, y, 50);
        }
    }

    add(&src1, &src2, &dst).unwrap();

    let dst_data = dst.data();

    // 100 + 50 = 150
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst_data[y * width + x],
                150,
                "Addition failed at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_add_saturation() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with values that would overflow
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 200);
            src2.set_pixel(x, y, 100);
        }
    }

    add(&src1, &src2, &dst).unwrap();

    let dst_data = dst.data();

    // 200 + 100 should saturate to 255
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst_data[y * width + x],
                255,
                "Saturation failed at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_subtract_basic() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with test values
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 150);
            src2.set_pixel(x, y, 50);
        }
    }

    subtract(&src1, &src2, &dst).unwrap();

    let dst_data = dst.data();

    // 150 - 50 = 100
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst_data[y * width + x],
                100,
                "Subtraction failed at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_subtract_floor() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with values that would underflow
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 50);
            src2.set_pixel(x, y, 100);
        }
    }

    subtract(&src1, &src2, &dst).unwrap();

    let dst_data = dst.data();

    // 50 - 100 should floor to 0
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst_data[y * width + x],
                0,
                "Flooring failed at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_multiply_basic() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with test values
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 128);
            src2.set_pixel(x, y, 128);
        }
    }

    // Scale = 1.0
    multiply(&src1, &src2, &dst, 1.0).unwrap();

    let dst_data = dst.data();

    // 128 * 128 * 1.0 / 255 ≈ 64
    for y in 0..height {
        for x in 0..width {
            let expected = ((128u32 * 128) / 255) as u8;
            let actual = dst_data[y * width + x];
            assert!(
                (actual as i16 - expected as i16).abs() <= 2,
                "Multiplication failed at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_weighted_average() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with test values
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 200);
            src2.set_pixel(x, y, 100);
        }
    }

    // alpha = 0.5 means equal weighting
    weighted(&src1, &src2, &dst, 0.5).unwrap();

    let dst_data = dst.data();

    // (200 * 128 + 100 * 128) / 256 = 150 (or 149 due to rounding)
    for y in 0..height {
        for x in 0..width {
            let expected = 150;
            let actual = dst_data[y * width + x];
            // Allow small difference due to integer rounding
            let diff = (actual as i16 - expected as i16).abs();
            assert!(
                diff <= 1,
                "Weighted average failed at ({}, {}): expected {}, got {}",
                x,
                y,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_weighted_full_alpha() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // Fill with different values
    for y in 0..height {
        for x in 0..width {
            src1.set_pixel(x, y, 200);
            src2.set_pixel(x, y, 100);
        }
    }

    // alpha ~1.0 means src1 gets all weight (almost)
    weighted(&src1, &src2, &dst, 1.0).unwrap();

    let dst_data = dst.data();

    // Should be close to src1 value
    for y in 0..height {
        for x in 0..width {
            let actual = dst_data[y * width + x];
            assert!(
                actual > 195,
                "Alpha=255 should give nearly full src1 weight: got {}",
                actual
            );
        }
    }
}

#[test]
fn test_min_image_basic() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    // src1: lower at (x>=y), src2: lower at (x<y)
    for y in 0..height {
        for x in 0..width {
            if x >= y {
                src1.set_pixel(x, y, 0x10);
                src2.set_pixel(x, y, 0x11);
            } else {
                src1.set_pixel(x, y, 0x11);
                src2.set_pixel(x, y, 0x10);
            }
        }
    }

    min_image(&src1, &src2, &dst).unwrap();

    let dst_data = dst.data();
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst_data[y * width + x],
                0x10,
                "Min should be 0x10 at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_max_image_basic() {
    let width = 4;
    let height = 4;

    let src1 = Image::new(width, height, ImageFormat::Gray);
    let src2 = Image::new(width, height, ImageFormat::Gray);
    let dst = Image::new(width, height, ImageFormat::Gray);

    for y in 0..height {
        for x in 0..width {
            if x >= y {
                src1.set_pixel(x, y, 0x10);
                src2.set_pixel(x, y, 0x11);
            } else {
                src1.set_pixel(x, y, 0x11);
                src2.set_pixel(x, y, 0x10);
            }
        }
    }

    max_image(&src1, &src2, &dst).unwrap();

    let dst_data = dst.data();
    for y in 0..height {
        for x in 0..width {
            assert_eq!(
                dst_data[y * width + x],
                0x11,
                "Max should be 0x11 at ({}, {})",
                x,
                y
            );
        }
    }
}

#[test]
fn test_min_max_image_dim_mismatch() {
    let a = Image::new(4, 4, ImageFormat::Gray);
    let b = Image::new(8, 8, ImageFormat::Gray);
    let dst = Image::new(4, 4, ImageFormat::Gray);

    assert!(min_image(&a, &b, &dst).is_err());
    assert!(max_image(&a, &b, &dst).is_err());
}
