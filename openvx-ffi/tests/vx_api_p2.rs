//! Integration tests for the three `vx_api.h` functions implemented in
//! Phase-2 of the OpenVX 1.3.1 coverage plan
//! (`docs/openvx-1.3.1-coverage-plan.md`).
//!
//! Lives under `openvx-ffi/tests/` (rather than `openvx-core/tests/`) so
//! the binary links the full workspace just like the cdylib does — the
//! `vxReleaseImage` / `vxReleasePyramid` symbols `openvx-core` references
//! internally are otherwise unresolved when openvx-core is tested in
//! isolation.

use std::ffi::CString;
use std::sync::atomic::{AtomicU32, Ordering};

// Pull in the symbols from each workspace crate the cdylib aggregates
// so the linker keeps those rlibs on the test binary's link line —
// `openvx-core` has internal `extern "C"` declarations of
// `vxReleaseImage` / `vxReleasePyramid` (from openvx-image) and
// `vxReleaseArray` etc. (from openvx-buffer) which need those rlibs
// resolved at link time.
#[allow(unused_imports)]
use openvx_buffer::c_api::vxReleaseArray;
#[allow(unused_imports)]
use openvx_image::c_api::vxReleaseImage;

use openvx_core::c_api::{
    vxCreateContext, vxLoadKernels, vxReleaseContext, vxUnloadKernels, vx_context, vx_status,
    VX_ERROR_INVALID_PARAMETERS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_NOT_SUPPORTED, VX_SUCCESS,
};
use openvx_core::unified_c_api::{
    vxCreateGraph, vxRegisterKernelLibrary, vxReleaseGraph, vxSetGraphAttribute,
};

const VX_GRAPH_NUMNODES: i32 = 0x00080200;
const VX_GRAPH_PERFORMANCE: i32 = 0x00080202;
const VX_GRAPH_NUMPARAMETERS: i32 = 0x00080203;
const VX_GRAPH_STATE: i32 = 0x00080204;

// ---------------------------------------------------------------------------
// vxRegisterKernelLibrary + vxLoadKernels integration
// ---------------------------------------------------------------------------

static PUBLISH_CALLS: AtomicU32 = AtomicU32::new(0);
static UNPUBLISH_CALLS: AtomicU32 = AtomicU32::new(0);

unsafe extern "C" fn test_publish(_ctx: vx_context) -> vx_status {
    PUBLISH_CALLS.fetch_add(1, Ordering::SeqCst);
    VX_SUCCESS
}

unsafe extern "C" fn test_unpublish(_ctx: vx_context) -> vx_status {
    UNPUBLISH_CALLS.fetch_add(1, Ordering::SeqCst);
    VX_SUCCESS
}

#[test]
fn register_kernel_library_then_load_invokes_publish() {
    PUBLISH_CALLS.store(0, Ordering::SeqCst);
    UNPUBLISH_CALLS.store(0, Ordering::SeqCst);

    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    // Unique module name so this test doesn't collide with other tests
    // that may also poke the kernel-library registry.
    let module = CString::new("openvx-test.p2-register-library").unwrap();

    let status = unsafe {
        vxRegisterKernelLibrary(ctx, module.as_ptr(), Some(test_publish), Some(test_unpublish))
    };
    assert_eq!(status, VX_SUCCESS, "vxRegisterKernelLibrary should succeed");

    // Now load it — `publish` must fire.
    let status = vxLoadKernels(ctx, module.as_ptr());
    assert_eq!(
        status, VX_SUCCESS,
        "vxLoadKernels should succeed for a registered library"
    );
    assert_eq!(
        PUBLISH_CALLS.load(Ordering::SeqCst),
        1,
        "publish callback should have been invoked exactly once"
    );

    // Unloading must fire `unpublish`.
    let status = vxUnloadKernels(ctx, module.as_ptr());
    assert_eq!(status, VX_SUCCESS);
    assert_eq!(
        UNPUBLISH_CALLS.load(Ordering::SeqCst),
        1,
        "unpublish callback should have been invoked exactly once"
    );

    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn register_kernel_library_rejects_null_arguments() {
    let module = CString::new("does-not-matter").unwrap();

    // Null context.
    let s = unsafe {
        vxRegisterKernelLibrary(
            std::ptr::null_mut(),
            module.as_ptr(),
            Some(test_publish),
            Some(test_unpublish),
        )
    };
    assert_eq!(s, VX_ERROR_INVALID_REFERENCE);

    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    // Null module name.
    let s = unsafe {
        vxRegisterKernelLibrary(ctx, std::ptr::null(), Some(test_publish), Some(test_unpublish))
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    // Empty module name.
    let empty = CString::new("").unwrap();
    let s = unsafe {
        vxRegisterKernelLibrary(ctx, empty.as_ptr(), Some(test_publish), Some(test_unpublish))
    };
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn load_unregistered_module_returns_invalid_parameters() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());

    // A module name that nothing's registered for, AND that isn't one of
    // the hardcoded `openvx-core` / `openvx-vision` / `test-testmodule`
    // names. Must surface as INVALID_PARAMETERS, not silently succeed.
    let module = CString::new("definitely-not-a-real-module-name-zzz").unwrap();
    let s = vxLoadKernels(ctx, module.as_ptr());
    assert_eq!(s, VX_ERROR_INVALID_PARAMETERS);

    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

// ---------------------------------------------------------------------------
// vxSetGraphAttribute
// ---------------------------------------------------------------------------

#[test]
fn set_graph_attribute_rejects_null_graph() {
    let mut val: u32 = 0;
    let s = vxSetGraphAttribute(
        std::ptr::null_mut(),
        VX_GRAPH_NUMNODES,
        &mut val as *mut u32 as *const std::os::raw::c_void,
        std::mem::size_of::<u32>(),
    );
    assert_eq!(s, VX_ERROR_INVALID_REFERENCE);
}

#[test]
fn set_graph_attribute_returns_not_supported_for_spec_attributes() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());
    let graph = vxCreateGraph(ctx);
    assert!(!graph.is_null());

    // Every spec-defined graph attribute is runtime-derived state, so
    // writes must return VX_ERROR_NOT_SUPPORTED rather than silently
    // succeeding or mutating implementation-owned counters.
    for attr in [
        VX_GRAPH_NUMNODES,
        VX_GRAPH_PERFORMANCE,
        VX_GRAPH_NUMPARAMETERS,
        VX_GRAPH_STATE,
    ] {
        let mut val: u32 = 99;
        let s = vxSetGraphAttribute(
            graph,
            attr,
            &mut val as *mut u32 as *const std::os::raw::c_void,
            std::mem::size_of::<u32>(),
        );
        assert_eq!(
            s, VX_ERROR_NOT_SUPPORTED,
            "writing spec-defined graph attribute 0x{attr:08x} must report NOT_SUPPORTED"
        );
    }

    let mut g = graph;
    vxReleaseGraph(&mut g);
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}

#[test]
fn set_graph_attribute_returns_not_supported_for_unknown_attribute() {
    let ctx = vxCreateContext();
    assert!(!ctx.is_null());
    let graph = vxCreateGraph(ctx);
    assert!(!graph.is_null());

    let mut val: u32 = 1;
    let s = vxSetGraphAttribute(
        graph,
        // An attribute id well outside the Khronos graph range.
        0xDEAD_BEEFu32 as i32,
        &mut val as *mut u32 as *const std::os::raw::c_void,
        std::mem::size_of::<u32>(),
    );
    assert_eq!(s, VX_ERROR_NOT_SUPPORTED);

    let mut g = graph;
    vxReleaseGraph(&mut g);
    let mut c = ctx;
    vxReleaseContext(&mut c as *mut vx_context);
}
