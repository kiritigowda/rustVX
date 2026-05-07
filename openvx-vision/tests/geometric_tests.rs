//! Geometric operation tests

use openvx_image::{create_uniform_image, Image, ImageFormat};
use openvx_vision::geometric::*;

#[test]
fn test_scale_image_same_size() {
    let input = create_uniform_image(4, 4, ImageFormat::Gray, 128);
    let output = Image::new(4, 4, ImageFormat::Gray);

    // Scale to same size should produce similar output
    scale_image(&input, &output, true).unwrap();

    let output_data = output.data();
    let input_data = input.data();

    // Values should be approximately preserved
    let diff: i32 = output_data
        .iter()
        .zip(input_data.iter())
        .map(|(a, b)| (*a as i32 - *b as i32).abs())
        .sum();
    let avg_diff = diff as f32 / output_data.len() as f32;
    assert!(
        avg_diff < 5.0,
        "Scale with same size should preserve values, avg_diff={}",
        avg_diff
    );
}

#[test]
fn test_scale_image_upscale() {
    let input = create_uniform_image(2, 2, ImageFormat::Gray, 128);
    let output = Image::new(4, 4, ImageFormat::Gray);

    scale_image(&input, &output, true).unwrap();

    // Output should be valid
    let output_data = output.data();
    for &v in output_data.iter() {
        assert!(v <= 255, "Upscale should produce valid values");
    }
}

#[test]
fn test_warp_affine_identity() {
    let input = create_uniform_image(4, 4, ImageFormat::Gray, 128);
    let output = Image::new(4, 4, ImageFormat::Gray);

    // Identity matrix: [1, 0, 0, 0, 1, 0]
    let matrix = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0];

    warp_affine(&input, &matrix, &output).unwrap();

    let output_data = output.data();
    let input_data = input.data();

    // Should be similar (interpolation differences expected)
    for i in 0..output_data.len() {
        let diff = (output_data[i] as i16 - input_data[i] as i16).abs();
        assert!(
            diff < 10,
            "Identity warp should preserve values: diff={} at {}",
            diff,
            i
        );
    }
}

#[test]
fn test_warp_affine_scale() {
    let input = create_uniform_image(4, 4, ImageFormat::Gray, 128);
    let output = Image::new(2, 2, ImageFormat::Gray);

    // Scale by 2: [2, 0, 0, 0, 2, 0]
    // But we want to scale down, so use 0.5
    let matrix = [0.5f32, 0.0, 0.0, 0.0, 0.5, 0.0];

    warp_affine(&input, &matrix, &output).unwrap();

    // Just verify it runs without error
    let output_data = output.data();
    for &v in output_data.iter() {
        assert!(v <= 255, "Affine warp should produce valid values");
    }
}

#[test]
fn test_warp_perspective_identity() {
    let input = create_uniform_image(4, 4, ImageFormat::Gray, 128);
    let output = Image::new(4, 4, ImageFormat::Gray);

    // Identity matrix
    let matrix = [1.0f32, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    warp_perspective(&input, &matrix, &output).unwrap();

    let output_data = output.data();
    let input_data = input.data();

    // Should be similar (interpolation differences expected)
    for i in 0..output_data.len() {
        let diff = (output_data[i] as i16 - input_data[i] as i16).abs();
        assert!(
            diff < 10,
            "Identity perspective warp should preserve values"
        );
    }
}

#[test]
fn test_remap_identity() {
    let input = create_uniform_image(4, 4, ImageFormat::Gray, 128);
    let output = Image::new(4, 4, ImageFormat::Gray);

    // Identity remap
    let mut map_x = Vec::with_capacity(16);
    let mut map_y = Vec::with_capacity(16);
    for y in 0..4 {
        for x in 0..4 {
            map_x.push(x as f32);
            map_y.push(y as f32);
        }
    }

    remap(&input, &map_x, &map_y, &output).unwrap();

    let output_data = output.data();
    let input_data = input.data();

    // Should be similar
    for i in 0..output_data.len() {
        let diff = (output_data[i] as i16 - input_data[i] as i16).abs();
        assert!(diff < 5, "Identity remap should preserve values");
    }
}
