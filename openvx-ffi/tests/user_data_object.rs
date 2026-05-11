//! Integration tests for the OpenVX 1.3.1 User Data Object data type
//! (`include/VX/vx_khr_user_data_object.h`). Covers the full 7-function
//! surface: create + virtual + release + query + copy + map + unmap.
//!
//! Lives under `openvx-ffi/tests/` (rather than `openvx-buffer/tests/`)
//! so the test binary picks up every workspace rlib — same rationale as
//! `openvx-ffi/tests/vx_api_p2.rs`.

use std::ffi::CString;

// Force the workspace's rlibs onto this test binary's link line —
// `openvx-core` has internal `extern "C"` references back to symbols
// defined in `openvx-image` / `openvx-buffer` that the linker would
// otherwise leave unresolved when the test only imports openvx-core.
#[allow(unused_imports)]
use openvx_buffer::c_api::vxReleaseArray;
#[allow(unused_imports)]
use openvx_image::c_api::vxReleaseImage;

use openvx_buffer::user_data_object::{
    vxCopyUserDataObject, vxCreateUserDataObject, vxCreateVirtualUserDataObject, vxMapUserDataObject,
    vxQueryUserDataObject, vxReleaseUserDataObject, vxUnmapUserDataObject, vx_user_data_object,
    VX_MAX_REFERENCE_NAME, VX_USER_DATA_OBJECT_NAME, VX_USER_DATA_OBJECT_SIZE,
};
use openvx_core::c_api::{
    vxCreateContext, vxReleaseContext, vx_context, vx_map_id, vx_size,
    VX_ERROR_INVALID_PARAMETERS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_NOT_SUPPORTED,
    VX_MEMORY_TYPE_HOST, VX_READ_AND_WRITE, VX_READ_ONLY, VX_SUCCESS, VX_WRITE_ONLY,
};
use openvx_core::unified_c_api::{vxCreateGraph, vxReleaseGraph};

// ---------------------------------------------------------------------------
// Lifecycle: create / virtual / release
// ---------------------------------------------------------------------------

#[test]
fn create_with_initial_bytes_round_trips_through_query_and_copy() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    let type_name = CString::new("widget").unwrap();
    let initial: [u8; 16] = [
        0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E,
        0x1F,
    ];

    let udo = unsafe {
        vxCreateUserDataObject(
            ctx,
            type_name.as_ptr(),
            initial.len() as vx_size,
            initial.as_ptr() as *const _,
        )
    };
    assert!(!udo.is_null());

    // Query the size attribute.
    let mut got_size: vx_size = 0;
    let s = unsafe {
        vxQueryUserDataObject(
            udo,
            VX_USER_DATA_OBJECT_SIZE,
            &mut got_size as *mut _ as *mut _,
            std::mem::size_of::<vx_size>(),
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert_eq!(got_size, initial.len() as vx_size);

    // Query the type-name attribute.
    let mut name_buf = [0u8; VX_MAX_REFERENCE_NAME];
    let s = unsafe {
        vxQueryUserDataObject(
            udo,
            VX_USER_DATA_OBJECT_NAME,
            name_buf.as_mut_ptr() as *mut _,
            VX_MAX_REFERENCE_NAME,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    // Trim at first NUL and convert.
    let trimmed = &name_buf[..name_buf.iter().position(|&b| b == 0).unwrap_or(0)];
    assert_eq!(std::str::from_utf8(trimmed).unwrap(), "widget");

    // Copy the bytes back out and verify they match the initial input.
    let mut readback = [0u8; 16];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            0,
            readback.len() as vx_size,
            readback.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert_eq!(readback, initial);

    // Release.
    let mut h = udo;
    let s = unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    assert_eq!(s, VX_SUCCESS);
    assert!(h.is_null(), "release must zero the caller's pointer");

    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn create_with_null_type_name_returns_empty_string_via_query() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    // Per spec, type_name=NULL → query returns empty string.
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 0, std::ptr::null()) };
    assert!(!udo.is_null());

    let mut name_buf = [0u8; VX_MAX_REFERENCE_NAME];
    let s = unsafe {
        vxQueryUserDataObject(
            udo,
            VX_USER_DATA_OBJECT_NAME,
            name_buf.as_mut_ptr() as *mut _,
            VX_MAX_REFERENCE_NAME,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert_eq!(name_buf[0], 0u8, "first byte must be NUL terminator");

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn create_with_null_initial_zeroes_the_buffer() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    let type_name = CString::new("zeroed").unwrap();
    let udo = unsafe { vxCreateUserDataObject(ctx, type_name.as_ptr(), 32, std::ptr::null()) };
    assert!(!udo.is_null());

    let mut buf = [0xFFu8; 32];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            0,
            buf.len() as vx_size,
            buf.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert!(buf.iter().all(|&b| b == 0), "buffer must be zero-initialised");

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn create_with_oversized_type_name_returns_null() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    // The spec caps `type_name` at VX_MAX_REFERENCE_NAME-1 bytes
    // (the buffer is `vx_char[VX_MAX_REFERENCE_NAME]` including NUL).
    let too_long = "a".repeat(VX_MAX_REFERENCE_NAME);
    let type_name = CString::new(too_long).unwrap();
    let udo = unsafe { vxCreateUserDataObject(ctx, type_name.as_ptr(), 1, std::ptr::null()) };
    assert!(udo.is_null(), "oversized type_name must reject");

    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn create_virtual_disallows_copy_and_map() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());
    let graph = vxCreateGraph(ctx);
    assert!(!graph.is_null());

    let type_name = CString::new("virt").unwrap();
    let udo = unsafe { vxCreateVirtualUserDataObject(graph, type_name.as_ptr(), 8) };
    assert!(!udo.is_null());

    // Virtual UDOs may not be accessed by the application.
    let mut buf = [0u8; 8];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            0,
            buf.len() as vx_size,
            buf.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_ne!(s, VX_SUCCESS, "copy on a virtual UDO must fail");

    let mut map_id: vx_map_id = 0;
    let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let s = unsafe {
        vxMapUserDataObject(
            udo,
            0,
            buf.len() as vx_size,
            &mut map_id,
            &mut ptr,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
            0,
        )
    };
    assert_ne!(s, VX_SUCCESS, "map on a virtual UDO must fail");

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut g = graph;
    vxReleaseGraph(&mut g);
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn release_null_pointer_returns_invalid_reference() {
    let s = unsafe { vxReleaseUserDataObject(std::ptr::null_mut()) };
    assert_eq!(s, VX_ERROR_INVALID_REFERENCE);

    let mut nullp: vx_user_data_object = std::ptr::null_mut();
    let s = unsafe { vxReleaseUserDataObject(&mut nullp as *mut vx_user_data_object) };
    assert_eq!(s, VX_ERROR_INVALID_REFERENCE);
}

// ---------------------------------------------------------------------------
// vxQueryUserDataObject — attribute validation
// ---------------------------------------------------------------------------

#[test]
fn query_unknown_attribute_returns_not_supported() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 4, std::ptr::null()) };
    assert!(!udo.is_null());

    let mut v: u32 = 0;
    let s = unsafe {
        vxQueryUserDataObject(
            udo,
            0xDEAD_BEEFu32 as i32,
            &mut v as *mut _ as *mut _,
            std::mem::size_of::<u32>(),
        )
    };
    assert_eq!(s, VX_ERROR_NOT_SUPPORTED);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn query_with_wrong_buffer_size_returns_invalid_parameters() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 16, std::ptr::null()) };

    // Size attribute requires sizeof(vx_size) buffer.
    let mut v: u8 = 0;
    let s = unsafe {
        vxQueryUserDataObject(
            udo,
            VX_USER_DATA_OBJECT_SIZE,
            &mut v as *mut _ as *mut _,
            1,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    // Name attribute requires exactly VX_MAX_REFERENCE_NAME bytes.
    let mut name_buf = [0u8; 8];
    let s = unsafe {
        vxQueryUserDataObject(
            udo,
            VX_USER_DATA_OBJECT_NAME,
            name_buf.as_mut_ptr() as *mut _,
            name_buf.len() as vx_size,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

// ---------------------------------------------------------------------------
// vxCopyUserDataObject — bounds checks and read/write
// ---------------------------------------------------------------------------

#[test]
fn copy_write_then_copy_read_round_trip_at_offset() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 32, std::ptr::null()) };

    let write_payload: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            8,
            4,
            write_payload.as_ptr() as *mut _,
            VX_WRITE_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_SUCCESS);

    let mut readback = [0u8; 4];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            8,
            4,
            readback.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert_eq!(readback, write_payload);

    // size==0 means "copy until end of object". Read the trailing 24 bytes.
    let mut tail = [0xFFu8; 24];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            8,
            0,
            tail.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert_eq!(&tail[..4], &write_payload);
    assert!(tail[4..].iter().all(|&b| b == 0));

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn copy_out_of_bounds_returns_invalid_parameters() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 16, std::ptr::null()) };
    let mut buf = [0u8; 32];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            10, // offset
            10, // size — extends past end (10 + 10 > 16)
            buf.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

// ---------------------------------------------------------------------------
// vxMapUserDataObject / vxUnmapUserDataObject
// ---------------------------------------------------------------------------

#[test]
fn map_unmap_round_trip_writes_through_to_storage() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 16, std::ptr::null()) };

    let mut map_id: vx_map_id = 0;
    let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let s = unsafe {
        vxMapUserDataObject(
            udo,
            4,
            8,
            &mut map_id,
            &mut ptr,
            VX_READ_AND_WRITE,
            VX_MEMORY_TYPE_HOST,
            0,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    assert!(!ptr.is_null());

    // Write through the mapped pointer.
    unsafe {
        let bytes = std::slice::from_raw_parts_mut(ptr as *mut u8, 8);
        for (i, b) in bytes.iter_mut().enumerate() {
            *b = (0x40 + i) as u8;
        }
    }

    let s = unsafe { vxUnmapUserDataObject(udo, map_id) };
    assert_eq!(s, VX_SUCCESS);

    // The writes should be observable via a separate copy-read.
    let mut readback = [0u8; 8];
    let s = unsafe {
        vxCopyUserDataObject(
            udo,
            4,
            8,
            readback.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_SUCCESS);
    let expected: [u8; 8] = [0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47];
    assert_eq!(readback, expected);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn unmap_with_unknown_id_returns_invalid_parameters() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 8, std::ptr::null()) };

    let s = unsafe { vxUnmapUserDataObject(udo, 9999) };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn map_out_of_bounds_returns_invalid_parameters() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 8, std::ptr::null()) };

    let mut map_id: vx_map_id = 0;
    let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();
    let s = unsafe {
        vxMapUserDataObject(
            udo,
            6,
            5, // 6 + 5 > 8
            &mut map_id,
            &mut ptr,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
            0,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn map_with_invalid_usage_or_mem_type_returns_invalid_parameters() {
    let ctx = vxCreateContext();
    let udo = unsafe { vxCreateUserDataObject(ctx, std::ptr::null(), 8, std::ptr::null()) };

    let mut map_id: vx_map_id = 0;
    let mut ptr: *mut std::os::raw::c_void = std::ptr::null_mut();

    // Invalid usage.
    let s = unsafe {
        vxMapUserDataObject(
            udo,
            0,
            8,
            &mut map_id,
            &mut ptr,
            0x12345678,
            VX_MEMORY_TYPE_HOST,
            0,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    // Invalid memory type.
    let s = unsafe {
        vxMapUserDataObject(
            udo,
            0,
            8,
            &mut map_id,
            &mut ptr,
            VX_READ_ONLY,
            0x12345678,
            0,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut h = udo;
    unsafe { vxReleaseUserDataObject(&mut h as *mut vx_user_data_object) };
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

// ---------------------------------------------------------------------------
// Cross-type pointer-confusion guard
// ---------------------------------------------------------------------------

#[test]
fn operations_on_non_udo_pointer_return_invalid_reference() {
    let ctx = vxCreateContext();

    // A fake pointer that was never returned from vxCreateUserDataObject.
    let fake: vx_user_data_object = 0xDEAD_BEEFusize as _;
    let mut got: vx_size = 0;
    let s = unsafe {
        vxQueryUserDataObject(
            fake,
            VX_USER_DATA_OBJECT_SIZE,
            &mut got as *mut _ as *mut _,
            std::mem::size_of::<vx_size>(),
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_REFERENCE);

    let mut buf = [0u8; 4];
    let s = unsafe {
        vxCopyUserDataObject(
            fake,
            0,
            4,
            buf.as_mut_ptr() as *mut _,
            VX_READ_ONLY,
            VX_MEMORY_TYPE_HOST,
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_REFERENCE);

    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}
