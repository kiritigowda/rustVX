import re

with open('openvx-core/src/vxu_impl.rs', 'r') as f:
    content = f.read()

old_phase = '''fn phase_s16(grad_x: &Image, grad_y: &Image, phase: &mut Image) {
    let width = grad_x.width();
    let height = grad_x.height();
    let phase_data = phase.data_mut();

    for y in 0..height {
        for x in 0..width {
            let gx = grad_x.get_pixel_s16(x, y) as f64;
            let gy = grad_y.get_pixel_s16(x, y) as f64;

            // CTS reference: atan2(gy, gx) * 256 / (M_PI * 2)
            let mut val = gy.atan2(gx) * 256.0 / (std::f64::consts::PI * 2.0);
            if val < 0.0 {
                val += 256.0;
            }
            let mut ival = (val + 0.5).floor() as i32;
            if ival >= 256 {
                ival -= 256;
            }
            let idx = y * width + x;
            if let Some(p) = phase_data.get_mut(idx) {
                *p = ival.clamp(0, 255) as u8;
            }
        }
    }
}'''

new_phase = '''fn phase_s16(grad_x: &Image, grad_y: &Image, phase: &mut Image) {
    let width = grad_x.width();
    let height = grad_y.height();
    let pixels = width * height;
    let phase_data = phase.data_mut();

    // Fast path: direct slice access for S16 data (little-endian)
    let gx_data = grad_x.data();
    let gy_data = grad_y.data();

    if gx_data.len() >= pixels * 2 && gy_data.len() >= pixels * 2 && phase_data.len() >= pixels {
        crate::kernel_fast_paths::phase_s16_fast(
            &gx_data[..pixels * 2],
            &gy_data[..pixels * 2],
            &mut phase_data[..pixels],
            pixels,
        );
    } else {
        // Fallback: pixel-by-pixel with bounds checks
        for y in 0..height {
            for x in 0..width {
                let gx = grad_x.get_pixel_s16(x, y) as f64;
                let gy = grad_y.get_pixel_s16(x, y) as f64;
                let mut val = gy.atan2(gx) * 256.0 / (std::f64::consts::PI * 2.0);
                if val < 0.0 { val += 256.0; }
                let mut ival = (val + 0.5).floor() as i32;
                if ival >= 256 { ival -= 256; }
                let idx = y * width + x;
                if let Some(p) = phase_data.get_mut(idx) {
                    *p = ival.clamp(0, 255) as u8;
                }
            }
        }
    }
}'''

if old_phase in content:
    content = content.replace(old_phase, new_phase)
    with open('openvx-core/src/vxu_impl.rs', 'w') as f:
        f.write(content)
    print("Replaced phase_s16 successfully")
else:
    print("Could not find old phase_s16")
    # Debug: show what we're looking for
    print("Looking for:")
    print(repr(old_phase[:100]))
