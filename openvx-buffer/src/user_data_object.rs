//! C API for the OpenVX 1.3.1 **User Data Object** extension
//! ([`include/VX/vx_khr_user_data_object.h`]).
//!
//! A User Data Object (UDO) is a strongly-typed binary blob — a `Vec<u8>`
//! tagged with a short type-name string — that the OpenVX graph can carry
//! between user-defined kernels. It's the data type the **Classifier**
//! extension (vx_khr_class.h) uses to pass classifier models, and that
//! NN code uses to pass weight / config tensors that don't fit the
//! `vx_tensor` shape.
//!
//! ## Why this lives in `openvx-buffer`
//!
//! The data type has the same shape as `vx_array` (variable-size blob,
//! reference-counted, with `Map` / `Unmap` semantics), and the existing
//! `vx_array` plumbing in this crate gives us a tested pattern for the
//! `REFERENCE_COUNTS` / `REFERENCE_TYPES` integration. Putting UDO here
//! keeps all "raw buffer-like" data types together.
//!
//! ## API coverage
//!
//! Implements all 7 functions from `vx_khr_user_data_object.h`:
//!   * `vxCreateUserDataObject`
//!   * `vxCreateVirtualUserDataObject`
//!   * `vxReleaseUserDataObject`
//!   * `vxQueryUserDataObject`
//!   * `vxCopyUserDataObject`
//!   * `vxMapUserDataObject`
//!   * `vxUnmapUserDataObject`
//!
//! Tracked by `docs/openvx-1.3.1-coverage-plan.md` (P2 phase, issues
//! #19-#22).

#![allow(non_camel_case_types)]
#![allow(dead_code)]

use openvx_core::c_api::{
    vx_char, vx_context, vx_enum, vx_graph, vx_map_id, vx_size, vx_status, vx_uint32,
    VX_ERROR_INVALID_PARAMETERS, VX_ERROR_INVALID_REFERENCE, VX_ERROR_NOT_SUPPORTED,
    VX_MEMORY_TYPE_HOST, VX_MEMORY_TYPE_NONE, VX_READ_AND_WRITE, VX_READ_ONLY, VX_SUCCESS,
    VX_WRITE_ONLY,
};
use openvx_core::unified_c_api::{REFERENCE_COUNTS, REFERENCE_TYPES};
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    RwLock,
};

// ---------------------------------------------------------------------------
// Spec constants
//
// These could move into `openvx-core::c_api` once UDO is fully wired in
// (it'd be more "central"). Kept here in `user_data_object.rs` for now so
// the UDO implementation is a single self-contained file.
// ---------------------------------------------------------------------------

/// `vx_user_data_object` — opaque handle, layout-compatible with the
/// `typedef struct _vx_user_data_object * vx_user_data_object;` declared
/// in `vx_khr_user_data_object.h`.
pub type vx_user_data_object = *mut c_void;

/// `VX_TYPE_USER_DATA_OBJECT` — object type enum used to register UDOs in
/// the unified `REFERENCE_TYPES` registry alongside arrays, images, etc.
pub const VX_TYPE_USER_DATA_OBJECT: vx_enum = 0x816;

/// `VX_USER_DATA_OBJECT_NAME` — read-only attribute returning the
/// `vx_char[VX_MAX_REFERENCE_NAME]` type-name string the UDO was created
/// with. Computed as `VX_ATTRIBUTE_BASE(VX_ID_KHRONOS=0, 0x816) + 0`.
pub const VX_USER_DATA_OBJECT_NAME: vx_enum = 0x0008_1600;

/// `VX_USER_DATA_OBJECT_SIZE` — read-only attribute returning the
/// `vx_size` byte count of the UDO blob. `... + 1`.
pub const VX_USER_DATA_OBJECT_SIZE: vx_enum = 0x0008_1601;

/// `VX_MAX_REFERENCE_NAME` — spec-defined cap on the type-name string
/// length, including the NUL terminator. The OpenVX 1.3 spec sets this
/// to 64; we use the same value so an existing C caller's
/// `vx_char[VX_MAX_REFERENCE_NAME]` query buffer is always sized
/// correctly for our outputs.
pub const VX_MAX_REFERENCE_NAME: usize = 64;

// ---------------------------------------------------------------------------
// In-memory shape
// ---------------------------------------------------------------------------

/// Backing storage for one UDO. Boxed; `Box::into_raw()` produces the
/// opaque `vx_user_data_object` handle that we hand back to C.
///
/// The mapping table values are `(offset, size)` only — the bytes the
/// caller sees during a map live directly inside `data` (the same shape
/// as the spec's "subset of the user data object" wording), so no
/// staging buffer is allocated. This is simpler than `vx_array`'s
/// staging-buffer design because UDOs are byte-typed (no item-size
/// translation needed) and there is no concept of "active item count
/// vs capacity".
pub struct VxCUserDataObject {
    type_name: String,
    data: RwLock<Vec<u8>>,
    is_virtual: bool,
    mapped_ranges: RwLock<HashMap<vx_map_id, (vx_size, vx_size)>>,
}

static UDO_MAP_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// Common construction helper used by both the regular and virtual
/// creation paths. Returns a raw `vx_user_data_object` pointer on
/// success, or `null` on validation / allocation failure.
fn create_internal(
    type_name: *const vx_char,
    size: vx_size,
    initial: *const c_void,
    is_virtual: bool,
) -> vx_user_data_object {
    // The spec allows `size == 0`; the resulting UDO has an empty
    // backing buffer but is a valid handle (callers commonly pass
    // empty UDOs as marker arguments to user kernels).
    let name = if type_name.is_null() {
        String::new()
    } else {
        let raw = unsafe { CStr::from_ptr(type_name) };
        match raw.to_str() {
            Ok(s) if s.len() < VX_MAX_REFERENCE_NAME => s.to_string(),
            _ => {
                // Either non-UTF8 or longer than the spec's cap — the
                // spec mandates we reject longer names rather than
                // truncating, since callers will compare the name they
                // see back from `vxQueryUserDataObject` against what
                // they passed in.
                return std::ptr::null_mut();
            }
        }
    };

    // Initial bytes: copy from `initial` if non-null, otherwise zero-init.
    let mut bytes = vec![0u8; size];
    if !initial.is_null() && size > 0 {
        unsafe {
            std::ptr::copy_nonoverlapping(initial as *const u8, bytes.as_mut_ptr(), size);
        }
    }

    let udo = Box::new(VxCUserDataObject {
        type_name: name,
        data: RwLock::new(bytes),
        is_virtual,
        mapped_ranges: RwLock::new(HashMap::new()),
    });
    let raw = Box::into_raw(udo) as vx_user_data_object;
    register_handle(raw);
    raw
}

fn register_handle(raw: vx_user_data_object) {
    let addr = raw as usize;
    if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
        counts.insert(addr, AtomicUsize::new(1));
    }
    if let Ok(mut types) = REFERENCE_TYPES.lock() {
        types.insert(addr, VX_TYPE_USER_DATA_OBJECT);
    }
}

/// Convert a raw `vx_user_data_object` handle into a borrowed reference
/// after verifying it's registered as one. Used by every operation
/// past create/release to reject mistyped pointers (e.g. someone passed
/// a `vx_image` to `vxQueryUserDataObject`) without UB.
fn lookup<'a>(udo: vx_user_data_object) -> Option<&'a VxCUserDataObject> {
    if udo.is_null() {
        return None;
    }
    let addr = udo as usize;
    let is_udo = REFERENCE_TYPES
        .lock()
        .ok()
        .and_then(|t| t.get(&addr).copied())
        .map_or(false, |t| t == VX_TYPE_USER_DATA_OBJECT);
    if !is_udo {
        return None;
    }
    // SAFETY: the entry in `REFERENCE_TYPES` indicates we created this
    // pointer via `Box::into_raw(Box<VxCUserDataObject>)` and have not
    // yet freed it (release-without-other-references would have wiped
    // the entry). The borrow lifetime `'a` is intentional — callers
    // hand it back to C immediately, so the borrow is never observed
    // across a yield point.
    Some(unsafe { &*(udo as *const VxCUserDataObject) })
}

// ===========================================================================
// vxCreateUserDataObject
// ===========================================================================

/// Creates a reference to a User Data Object.
///
/// Spec: `vx_khr_user_data_object.h`:
///
/// ```c
/// vx_user_data_object vxCreateUserDataObject(
///     vx_context context,
///     const vx_char *type_name,
///     vx_size size,
///     const void *ptr);
/// ```
///
/// * `type_name == NULL` → empty type-name string (spec-mandated behaviour).
/// * `size == 0` → empty backing buffer; still a valid handle.
/// * `ptr == NULL` → buffer initialised to zeroes; otherwise `size`
///   bytes copied in from `ptr`.
///
/// # Safety
///
/// `type_name`, if non-null, must be a NUL-terminated C string. `ptr`,
/// if non-null, must point to at least `size` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn vxCreateUserDataObject(
    context: vx_context,
    type_name: *const vx_char,
    size: vx_size,
    ptr: *const c_void,
) -> vx_user_data_object {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    create_internal(type_name, size, ptr, false)
}

// ===========================================================================
// vxCreateVirtualUserDataObject
// ===========================================================================

/// Creates a virtual UDO scoped to a graph.
///
/// Spec: as above, but the resulting handle:
///
/// * Cannot be accessed by the application via `vxMap*` / `vxCopy*`
///   (those return `VX_ERROR_OPTIMIZED_AWAY`).
/// * Lives for as long as the parent graph lives.
///
/// We implement the same backing storage as the regular variant; the
/// only behavioural difference is the `is_virtual = true` flag, which
/// the access operations check.
#[no_mangle]
pub unsafe extern "C" fn vxCreateVirtualUserDataObject(
    graph: vx_graph,
    type_name: *const vx_char,
    size: vx_size,
) -> vx_user_data_object {
    if graph.is_null() {
        return std::ptr::null_mut();
    }
    create_internal(type_name, size, std::ptr::null(), true)
}

// ===========================================================================
// vxReleaseUserDataObject
// ===========================================================================

/// Releases a UDO reference and frees the backing storage when the
/// reference count drops to zero. Sets `*user_data_object` to NULL on
/// success (spec-mandated).
#[no_mangle]
pub unsafe extern "C" fn vxReleaseUserDataObject(
    user_data_object: *mut vx_user_data_object,
) -> vx_status {
    if user_data_object.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    let raw = *user_data_object;
    if raw.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    let addr = raw as usize;

    // Verify the pointer is actually one of ours.
    let is_udo = REFERENCE_TYPES
        .lock()
        .ok()
        .and_then(|t| t.get(&addr).copied())
        .map_or(false, |t| t == VX_TYPE_USER_DATA_OBJECT);
    if !is_udo {
        return VX_ERROR_INVALID_REFERENCE;
    }

    let should_free = REFERENCE_COUNTS.lock().ok().map_or(true, |counts| {
        if let Some(cnt) = counts.get(&addr) {
            let prev = cnt.fetch_sub(1, Ordering::SeqCst);
            prev <= 1
        } else {
            true
        }
    });

    if should_free {
        if let Ok(mut counts) = REFERENCE_COUNTS.lock() {
            counts.remove(&addr);
        }
        if let Ok(mut types) = REFERENCE_TYPES.lock() {
            types.remove(&addr);
        }
        // SAFETY: `should_free` only fires when the refcount was the
        // last outstanding one; we've already removed the registry
        // entries above, so no other code can observe this pointer.
        let _ = Box::from_raw(raw as *mut VxCUserDataObject);
    }

    *user_data_object = std::ptr::null_mut();
    VX_SUCCESS
}

// ===========================================================================
// vxQueryUserDataObject
// ===========================================================================

/// Queries the UDO for `VX_USER_DATA_OBJECT_NAME` (type-name string) or
/// `VX_USER_DATA_OBJECT_SIZE` (byte count).
///
/// The name is copied into the caller's `vx_char[VX_MAX_REFERENCE_NAME]`
/// buffer with NUL termination. The caller's `size` must be exactly
/// `VX_MAX_REFERENCE_NAME` (per spec — the buffer is a fixed-size
/// array, not a `size`-byte arbitrary buffer).
#[no_mangle]
pub unsafe extern "C" fn vxQueryUserDataObject(
    user_data_object: vx_user_data_object,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    let udo = match lookup(user_data_object) {
        Some(u) => u,
        None => return VX_ERROR_INVALID_REFERENCE,
    };
    if ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    match attribute {
        VX_USER_DATA_OBJECT_NAME => {
            if size != VX_MAX_REFERENCE_NAME {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            // Zero out the whole target buffer (gives a guaranteed NUL
            // terminator even if `type_name` is exactly
            // `VX_MAX_REFERENCE_NAME - 1` bytes long).
            std::ptr::write_bytes(ptr as *mut u8, 0, size);
            let bytes = udo.type_name.as_bytes();
            let n = bytes.len().min(size - 1);
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr as *mut u8, n);
            VX_SUCCESS
        }
        VX_USER_DATA_OBJECT_SIZE => {
            if size != std::mem::size_of::<vx_size>() {
                return VX_ERROR_INVALID_PARAMETERS;
            }
            let n = udo.data.read().unwrap().len();
            *(ptr as *mut vx_size) = n;
            VX_SUCCESS
        }
        _ => VX_ERROR_NOT_SUPPORTED,
    }
}

// ===========================================================================
// vxCopyUserDataObject
// ===========================================================================

/// Copies a `[offset, offset + size)` range between a UDO and user
/// memory, in either direction (`VX_READ_ONLY` = UDO → user,
/// `VX_WRITE_ONLY` = user → UDO).
#[no_mangle]
pub unsafe extern "C" fn vxCopyUserDataObject(
    user_data_object: vx_user_data_object,
    offset: vx_size,
    size: vx_size,
    user_ptr: *mut c_void,
    usage: vx_enum,
    user_mem_type: vx_enum,
) -> vx_status {
    let udo = match lookup(user_data_object) {
        Some(u) => u,
        None => return VX_ERROR_INVALID_REFERENCE,
    };
    if udo.is_virtual {
        // VX_ERROR_OPTIMIZED_AWAY = -23 in OpenVX 1.3.1; rustVX doesn't
        // export it as a constant yet, so we surface it as INVALID_REFERENCE
        // (the closest match a user can already react to). A follow-up
        // P8 cleanup can add the proper error code and tighten this.
        return VX_ERROR_INVALID_REFERENCE;
    }
    if user_ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if user_mem_type != VX_MEMORY_TYPE_HOST && user_mem_type != VX_MEMORY_TYPE_NONE {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    // Per spec: `size == 0` means "copy until the end of the object".
    let buf_len = udo.data.read().unwrap().len();
    let effective_size = if size == 0 {
        buf_len.saturating_sub(offset)
    } else {
        size
    };
    if offset.saturating_add(effective_size) > buf_len {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if effective_size == 0 {
        return VX_SUCCESS;
    }

    match usage {
        VX_READ_ONLY => {
            let data = udo.data.read().unwrap();
            std::ptr::copy_nonoverlapping(
                data.as_ptr().add(offset),
                user_ptr as *mut u8,
                effective_size,
            );
            VX_SUCCESS
        }
        VX_WRITE_ONLY => {
            let mut data = udo.data.write().unwrap();
            std::ptr::copy_nonoverlapping(
                user_ptr as *const u8,
                data.as_mut_ptr().add(offset),
                effective_size,
            );
            VX_SUCCESS
        }
        _ => VX_ERROR_INVALID_PARAMETERS,
    }
}

// ===========================================================================
// vxMapUserDataObject / vxUnmapUserDataObject
// ===========================================================================

/// Provides direct pointer access to a `[offset, offset + size)` range
/// of the UDO's bytes for in-place inspection / modification, in
/// preference to copying through a user buffer. Pair with
/// [`vxUnmapUserDataObject`] to finalise the access.
///
/// Because the UDO's `data: Vec<u8>` is a single contiguous allocation
/// that never reallocates after creation, we can hand the caller a
/// pointer directly into that allocation rather than maintaining a
/// staging buffer like `vxMapArrayRange` does.
#[no_mangle]
pub unsafe extern "C" fn vxMapUserDataObject(
    user_data_object: vx_user_data_object,
    offset: vx_size,
    size: vx_size,
    map_id: *mut vx_map_id,
    ptr: *mut *mut c_void,
    usage: vx_enum,
    mem_type: vx_enum,
    _flags: vx_uint32,
) -> vx_status {
    let udo = match lookup(user_data_object) {
        Some(u) => u,
        None => return VX_ERROR_INVALID_REFERENCE,
    };
    if udo.is_virtual {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if map_id.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if mem_type != VX_MEMORY_TYPE_HOST && mem_type != VX_MEMORY_TYPE_NONE {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    if usage != VX_READ_ONLY && usage != VX_WRITE_ONLY && usage != VX_READ_AND_WRITE {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    let effective_size = {
        let data = udo.data.read().unwrap();
        let buf_len = data.len();
        let n = if size == 0 {
            buf_len.saturating_sub(offset)
        } else {
            size
        };
        if offset.saturating_add(n) > buf_len {
            return VX_ERROR_INVALID_PARAMETERS;
        }
        n
    };

    let id = UDO_MAP_ID_COUNTER.fetch_add(1, Ordering::SeqCst) as vx_map_id;

    // Take a write lock long enough to capture the raw byte pointer
    // — the underlying allocation never moves while the UDO is live,
    // so the pointer remains valid after we drop the guard, but the
    // borrow checker doesn't know that. We do the copy-out / direct-
    // pointer step under a fresh write lock.
    let raw_ptr = {
        let mut data = udo.data.write().unwrap();
        data.as_mut_ptr().add(offset) as *mut c_void
    };

    udo.mapped_ranges
        .write()
        .unwrap()
        .insert(id, (offset, effective_size));

    *map_id = id;
    *ptr = raw_ptr;
    VX_SUCCESS
}

/// Releases the lock established by [`vxMapUserDataObject`].
///
/// Because the map gives the caller a direct pointer into the UDO's
/// storage (no staging buffer), there is no "commit writes back" step
/// — any modifications the caller made are already visible. We simply
/// invalidate the `map_id` and drop the entry from the tracker so a
/// follow-up `vxMapUserDataObject` over the same range succeeds.
#[no_mangle]
pub unsafe extern "C" fn vxUnmapUserDataObject(
    user_data_object: vx_user_data_object,
    map_id: vx_map_id,
) -> vx_status {
    let udo = match lookup(user_data_object) {
        Some(u) => u,
        None => return VX_ERROR_INVALID_REFERENCE,
    };
    let removed = udo.mapped_ranges.write().unwrap().remove(&map_id);
    if removed.is_some() {
        VX_SUCCESS
    } else {
        VX_ERROR_INVALID_PARAMETERS
    }
}
