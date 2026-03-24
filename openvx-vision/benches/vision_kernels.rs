//! Performance benchmarks for vision kernels
//!
//! Run with: cargo bench --features "simd parallel"

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use openvx_image::{Image, ImageFormat};

fn create_test_image(width: usize, height: usize) -> Image {
    // Use saturating_mul to prevent integer overflow
    let data_size = width.saturating_mul(height);
    let mut data = vec![0u8; data_size];
    for (i, pixel) in data.iter_mut().enumerate() {
        *pixel = ((i * 17) & 0xFF) as u8;
    }
    Image::from_data(width, height, ImageFormat::Gray, data)
}

fn create_test_rgb(width: usize, height: usize) -> Image {
    // Use saturating_mul to prevent integer overflow
    let data_size = width.saturating_mul(height).saturating_mul(3);
    let mut data = vec![0u8; data_size];
    for (i, pixel) in data.iter_mut().enumerate() {
        *pixel = ((i * 13) & 0xFF) as u8;
    }
    Image::from_data(width, height, ImageFormat::Rgb, data)
}

// Gaussian 3x3 benchmarks
fn bench_gaussian3x3(c: &mut Criterion) {
    let sizes = vec![(640, 480), (1280, 720), (1920, 1080)];
    
    let mut group = c.benchmark_group("gaussian3x3");
    
    for (width, height) in &sizes {
        let src = create_test_image(*width, *height);
        let mut dst = Image::new(*width, *height, ImageFormat::Gray);
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::filter::gaussian3x3(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::filter_simd::gaussian3x3_simd(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::parallel::gaussian3x3_parallel(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// Box 3x3 benchmarks
fn bench_box3x3(c: &mut Criterion) {
    let sizes = vec![(640, 480), (1280, 720), (1920, 1080)];
    
    let mut group = c.benchmark_group("box3x3");
    
    for (width, height) in &sizes {
        let src = create_test_image(*width, *height);
        let mut dst = Image::new(*width, *height, ImageFormat::Gray);
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::filter::box3x3(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::filter_simd::box3x3_simd(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// RGB to Grayscale benchmarks
fn bench_rgb_to_gray(c: &mut Criterion) {
    let sizes = vec![(640, 480), (1280, 720), (1920, 1080)];
    
    let mut group = c.benchmark_group("rgb_to_gray");
    
    for (width, height) in &sizes {
        let src = create_test_rgb(*width, *height);
        let mut dst = Image::new(*width, *height, ImageFormat::Gray);
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::color::rgb_to_gray(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::color_simd::rgb_to_gray_simd(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::parallel::rgb_to_gray_parallel(
                        black_box(&src),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// Arithmetic benchmarks
fn bench_add_images(c: &mut Criterion) {
    let sizes = vec![(640, 480), (1280, 720), (1920, 1080)];
    
    let mut group = c.benchmark_group("add_images");
    
    for (width, height) in &sizes {
        let src1 = create_test_image(*width, *height);
        let src2 = create_test_image(*width, *height);
        let mut dst = Image::new(*width, *height, ImageFormat::Gray);
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::arithmetic::add(
                        black_box(&src1),
                        black_box(&src2),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::arithmetic_simd::add_images_simd(
                        black_box(&src1),
                        black_box(&src2),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::parallel::add_images_parallel(
                        black_box(&src1),
                        black_box(&src2),
                        black_box(&mut dst)
                    ).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// Sobel benchmark
fn bench_sobel3x3(c: &mut Criterion) {
    let sizes = vec![(640, 480), (1280, 720)];
    
    let mut group = c.benchmark_group("sobel3x3");
    
    for (width, height) in &sizes {
        let src = create_test_image(*width, *height);
        // Use saturating_mul to prevent integer overflow
        let buf_size = width.saturating_mul(*height);
        let mut gx = vec![0i16; buf_size];
        let mut gy = vec![0i16; buf_size];
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::gradient::sobel3x3_full(
                        black_box(&src)
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "parallel")]
        group.bench_with_input(
            BenchmarkId::new("parallel", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::parallel::sobel3x3_parallel(
                        black_box(&src),
                        black_box(&mut gx),
                        black_box(&mut gy)
                    ).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

// Weighted average benchmark
fn bench_weighted_average(c: &mut Criterion) {
    let sizes = vec![(640, 480), (1280, 720), (1920, 1080)];
    
    let mut group = c.benchmark_group("weighted_average");
    
    for (width, height) in &sizes {
        let src1 = create_test_image(*width, *height);
        let src2 = create_test_image(*width, *height);
        let mut dst = Image::new(*width, *height, ImageFormat::Gray);
        
        group.bench_with_input(
            BenchmarkId::new("scalar", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::arithmetic::weighted(
                        black_box(&src1),
                        black_box(&src2),
                        black_box(&mut dst),
                        128
                    ).unwrap()
                })
            }
        );
        
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd", format!("{}x{}", width, height)),
            &(*width, *height),
            |b, _| {
                b.iter(|| {
                    openvx_vision::arithmetic_simd::weighted_avg_simd(
                        black_box(&src1),
                        black_box(&src2),
                        black_box(&mut dst),
                        128
                    ).unwrap()
                })
            }
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_gaussian3x3,
    bench_box3x3,
    bench_rgb_to_gray,
    bench_add_images,
    bench_sobel3x3,
    bench_weighted_average
);
criterion_main!(benches);
