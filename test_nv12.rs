// Simple NV12 test
use std::sync::{RwLock, Arc};

fn main() {
    // Test NV12 size calculation
    let width = 1920u32;
    let height = 1080u32;
    
    // NV12: Y plane is full size, UV plane is half height (rounded up)
    let y_size = (width as usize) * (height as usize);
    let uv_height = (height + 1) / 2;  // Round up
    let uv_size = (width as usize) * (uv_height as usize);
    let total_size = y_size + uv_size;
    
    println!("NV12 size calculation for {}x{}:", width, height);
    println!("  Y plane size: {} ({}x{})", y_size, width, height);
    println!("  UV plane height: {} (rounded up from {})", uv_height, height);
    println!("  UV plane size: {} ({}x{})", uv_size, width, uv_height);
    println!("  Total size: {}", total_size);
    
    // Test odd height
    let height_odd = 1081u32;
    let y_size_odd = (width as usize) * (height_odd as usize);
    let uv_height_odd = (height_odd + 1) / 2;
    let uv_size_odd = (width as usize) * (uv_height_odd as usize);
    let total_size_odd = y_size_odd + uv_size_odd;
    
    println!("\nNV12 size calculation for {}x{}:", width, height_odd);
    println!("  Y plane size: {} ({}x{})", y_size_odd, width, height_odd);
    println!("  UV plane height: {} (rounded up from {})", uv_height_odd, height_odd);
    println!("  UV plane size: {} ({}x{})", uv_size_odd, width, uv_height_odd);
    println!("  Total size: {}", total_size_odd);
    
    // Test small size to verify allocation works
    let small_width = 10u32;
    let small_height = 5u32;
    let y_small = (small_width as usize) * (small_height as usize); // 50
    let uv_h_small = (small_height + 1) / 2; // 3
    let uv_small = (small_width as usize) * (uv_h_small as usize); // 30
    let total_small = y_small + uv_small; // 80
    
    println!("\nNV12 size calculation for {}x{}:", small_width, small_height);
    println!("  Y plane size: {} ({}x{})", y_small, small_width, small_height);
    println!("  UV plane height: {} (rounded up from {})", uv_h_small, small_height);
    println!("  UV plane size: {} ({}x{})", uv_small, small_width, uv_h_small);
    println!("  Total size: {}", total_small);
    
    // Test allocation
    let data = vec![0u8; total_small];
    println!("\n  Allocated {} bytes successfully!", data.len());
    
    // Simulate accessing UV plane at offset
    let uv_offset = y_small; // 50
    println!("  Y plane: bytes 0 to {}", y_small - 1);
    println!("  UV plane: bytes {} to {}", uv_offset, uv_offset + uv_small - 1);
    println!("  Buffer size: {}", data.len());
    
    // This would have been the bug before fix:
    // OLD: uv_size = w * (h / 2) = 10 * (5 / 2) = 10 * 2 = 20
    // NEW: uv_size = w * ((h + 1) / 2) = 10 * ((5 + 1) / 2) = 10 * 3 = 30
    let old_uv_size = (small_width as usize) * ((small_height / 2) as usize);
    println!("\n  BUG COMPARISON:");
    println!("  Old UV size (truncated): {}", old_uv_size);
    println!("  New UV size (rounded up): {}", uv_small);
    println!("  Difference: {} bytes", uv_small - old_uv_size);
}
