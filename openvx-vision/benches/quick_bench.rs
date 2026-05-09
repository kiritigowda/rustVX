//! Quick performance benchmarks for vision kernels (reduced iterations)

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use openvx_image::{Image, ImageFormat};

fn create_test_image(width: usize, height: usize) -> Image {
    let data_size = width.saturating_mul(height);
    let mut data = vec![0u8; data_size];
    for (i, pixel) in data.iter_mut().enumerate() {
        *pixel = ((i * 17) & 0xFF) as u8;
    }
    Image::from_data(width, height, ImageFormat::Gray, data)
}

fn create_test_rgb(width: usize, height: usize) -> Image {
    let data_size = width.saturating_mul(height).saturating_mul(3);
    let mut data = vec![0u8; data_size];
    for (i, pixel) in data.iter_mut().enumerate() {
        *pixel = ((i * 13) & 0xFF) as u8;
    }
    Image::from_data(width, height, ImageFormat::Rgb, data)
}

fn bench_box3x3(c: &mut Criterion) {
    let mut group = c.benchmark_group("box3x3");
    let (w, h) = (1920, 1080);
    let src = create_test_image(w, h);
    let mut dst = Image::new(w, h, ImageFormat::Gray);
    group.bench_function("scalar_1920x1080", |b| {
        b.iter(|| openvx_vision::filter::box3x3(black_box(&src), black_box(&mut dst)).unwrap())
    });
    #[cfg(feature = "simd")]
    group.bench_function("simd_1920x1080", |b| {
        b.iter(|| openvx_vision::filter_simd::box3x3_simd(black_box(&src), black_box(&mut dst)).unwrap())
    });
    group.finish();
}

fn bench_gaussian3x3(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian3x3");
    let (w, h) = (1920, 1080);
    let src = create_test_image(w, h);
    let mut dst = Image::new(w, h, ImageFormat::Gray);
    group.bench_function("scalar_1920x1080", |b| {
        b.iter(|| openvx_vision::filter::gaussian3x3(black_box(&src), black_box(&mut dst)).unwrap())
    });
    #[cfg(feature = "simd")]
    group.bench_function("simd_1920x1080", |b| {
        b.iter(|| openvx_vision::filter_simd::gaussian3x3_simd(black_box(&src), black_box(&mut dst)).unwrap())
    });
    group.finish();
}

fn bench_rgb_to_gray(c: &mut Criterion) {
    let mut group = c.benchmark_group("rgb_to_gray");
    let (w, h) = (1920, 1080);
    let src = create_test_rgb(w, h);
    let mut dst = Image::new(w, h, ImageFormat::Gray);
    group.bench_function("scalar_1920x1080", |b| {
        b.iter(|| openvx_vision::color::rgb_to_gray(black_box(&src), black_box(&mut dst)).unwrap())
    });
    group.finish();
}

fn bench_add_images(c: &mut Criterion) {
    let mut group = c.benchmark_group("add_images");
    let (w, h) = (1920, 1080);
    let src1 = create_test_image(w, h);
    let src2 = create_test_image(w, h);
    let mut dst = Image::new(w, h, ImageFormat::Gray);
    group.bench_function("scalar_1920x1080", |b| {
        b.iter(|| openvx_vision::arithmetic::add(black_box(&src1), black_box(&src2), black_box(&mut dst)).unwrap())
    });
    #[cfg(feature = "simd")]
    group.bench_function("simd_1920x1080", |b| {
        b.iter(|| openvx_vision::arithmetic_simd::add_images_simd(black_box(&src1), black_box(&src2), black_box(&mut dst)).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_box3x3, bench_gaussian3x3, bench_rgb_to_gray, bench_add_images);
criterion_main!(benches);
