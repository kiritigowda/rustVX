import re

with open('openvx-core/src/vxu_impl.rs', 'r') as f:
    content = f.read()

old_func = '''fn median3x3(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;

    let dst_data = dst.data_mut();
    let mut window = [0u8; 9];

    for y in 0..height {
        for x in 0..width {
            let mut idx = 0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let py = y as isize + dy;
                    let px = x as isize + dx;
                    if py >= 0 && py < height as isize && px >= 0 && px < width as isize {
                        window[idx] = src.get_pixel(px as usize, py as usize);
                    } else {
                        window[idx] = src.get_pixel(x, y); // Replicate border
                    }
                    idx += 1;
                }
            }

            let idx = y.saturating_mul(width).saturating_add(x);
            if let Some(p) = dst_data.get_mut(idx) {
                *p = quickselect(&mut window, 4);
            }
        }
    }

    Ok(())
}'''

new_func = '''fn median3x3(src: &Image, dst: &mut Image) -> VxResult<()> {
    let width = src.width;
    let height = src.height;
    let pixels = width * height;
    let dst_data = dst.data_mut();
    let src_data = src.data();

    if src_data.len() >= pixels && dst_data.len() >= pixels {
        crate::kernel_fast_paths::median3x3_u8_replicate(
            &src_data[..pixels], &mut dst_data[..pixels], width, height,
        );
    } else {
        let mut window = [0u8; 9];
        for y in 0..height {
            for x in 0..width {
                let mut idx = 0;
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let py = y as isize + dy;
                        let px = x as isize + dx;
                        if py >= 0 && py < height as isize && px >= 0 && px < width as isize {
                            window[idx] = src.get_pixel(px as usize, py as usize);
                        } else {
                            window[idx] = src.get_pixel(x, y); // Replicate border
                        }
                        idx += 1;
                    }
                }

                let idx = y.saturating_mul(width).saturating_add(x);
                if let Some(p) = dst_data.get_mut(idx) {
                    *p = quickselect(&mut window, 4);
                }
            }
        }
    }

    Ok(())
}'''

if old_func in content:
    content = content.replace(old_func, new_func)
    with open('openvx-core/src/vxu_impl.rs', 'w') as f:
        f.write(content)
    print("Replaced median3x3 successfully")
else:
    print("Could not find old median3x3")
