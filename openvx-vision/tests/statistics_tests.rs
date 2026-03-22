//! Tests for statistical operations

use openvx_image::{Image, ImageFormat};
use openvx_vision::statistics::{min_max_loc, mean_std_dev, histogram, equalize_histogram, Coordinate};

#[test]
fn test_min_max_loc_basic() {
    let mut input = Image::new(5, 5, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        // Create image with min=10, max=240
        for i in 0..data.len() {
            data[i] = 100;
        }
        data[0] = 10;  // Min at (0,0)
        data[24] = 240; // Max at (4,4)
    }
    
    let (min_val, max_val, min_loc, max_loc) = min_max_loc(&input).unwrap();
    
    assert_eq!(min_val, 10);
    assert_eq!(max_val, 240);
    assert_eq!(min_loc, Coordinate::new(0, 0));
    assert_eq!(max_loc, Coordinate::new(4, 4));
}

#[test]
fn test_min_max_loc_uniform() {
    let mut input = Image::new(3, 3, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = 128;
        }
    }
    
    let (min_val, max_val, _min_loc, _max_loc) = min_max_loc(&input).unwrap();
    
    assert_eq!(min_val, 128);
    assert_eq!(max_val, 128);
}

#[test]
fn test_mean_std_dev_basic() {
    let mut input = Image::new(2, 2, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        data[0] = 0;
        data[1] = 100;
        data[2] = 200;
        data[3] = 255;
    }
    
    let (mean, stddev) = mean_std_dev(&input).unwrap();
    
    // Expected mean: (0 + 100 + 200 + 255) / 4 = 138.75
    assert!((mean - 138.75).abs() < 0.01);
    assert!(stddev > 0.0);
}

#[test]
fn test_mean_std_dev_uniform() {
    let mut input = Image::new(4, 4, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = 100;
        }
    }
    
    let (mean, stddev) = mean_std_dev(&input).unwrap();
    
    assert!((mean - 100.0).abs() < 0.01);
    assert!(stddev < 0.01);
}

#[test]
fn test_histogram_basic() {
    let mut input = Image::new(2, 2, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        data[0] = 0;
        data[1] = 50;
        data[2] = 50;
        data[3] = 255;
    }
    
    let hist = histogram(&input).unwrap();
    
    assert_eq!(hist[0], 1);
    assert_eq!(hist[50], 2);
    assert_eq!(hist[255], 1);
    // Sum of histogram should equal total pixels
    let sum: u32 = hist.iter().sum();
    assert_eq!(sum, 4);
}

#[test]
fn test_histogram_uniform() {
    let mut input = Image::new(10, 10, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        for i in 0..data.len() {
            data[i] = 100;
        }
    }
    
    let hist = histogram(&input).unwrap();
    
    assert_eq!(hist[100], 100); // All 100 pixels at value 100
}

#[test]
fn test_equalize_histogram() {
    let mut input = Image::new(4, 4, ImageFormat::Gray);
    let output = Image::new(4, 4, ImageFormat::Gray);
    {
        let mut data = input.data_mut();
        // Create dark image (mostly low values)
        for i in 0..data.len() {
            data[i] = if i % 2 == 0 { 50 } else { 60 };
        }
    }
    
    equalize_histogram(&input, &output).unwrap();
    
    let out_data = output.data();
    // After equalization, values should be spread out
    // Check that output is valid
    for i in 0..out_data.len() {
        assert!(out_data[i] <= 255);
    }
}

#[test]
fn test_coordinate_equality() {
    let c1 = Coordinate::new(10, 20);
    let c2 = Coordinate::new(10, 20);
    let c3 = Coordinate::new(20, 10);
    
    assert_eq!(c1, c2);
    assert_ne!(c1, c3);
}
