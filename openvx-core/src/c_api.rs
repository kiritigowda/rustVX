//! C API for OpenVX Core
//!
//! This module provides FFI bindings for the OpenVX API

use std::ffi::{CStr, c_void};
use std::sync::{Arc, Mutex};
use crate::types::VxStatus;

// Import the unified CONTEXTS registry
use crate::unified_c_api::{CONTEXTS as UNIFIED_CONTEXTS, VxCContext};

// ============================================================================
// Type Aliases (C-compatible)
// ============================================================================

/// Basic OpenVX types
pub type vx_status = i32;
pub type vx_enum = i32;
pub type vx_uint32 = u32;
pub type vx_size = usize;
pub type vx_char = i8;
pub type vx_bool = i32;
pub type vx_df_image = u32;
pub type vx_uint64 = u64;
pub type vx_int32 = i32;
pub type vx_float32 = f32;
pub type vx_map_id = usize;

// ============================================================================
// Forward Declarations for Opaque Types
// ============================================================================

/// Opaque reference to a context
pub enum VxContext {}
pub type vx_context = *mut VxContext;

/// Opaque reference to a graph
pub enum VxGraph {}
pub type vx_graph = *mut VxGraph;

/// Opaque reference to a node
pub enum VxNode {}
pub type vx_node = *mut VxNode;

/// Opaque reference to a kernel
pub enum VxKernel {}
pub type vx_kernel = *mut VxKernel;

/// Opaque reference to a parameter
pub enum VxParameter {}
pub type vx_parameter = *mut VxParameter;

/// Opaque reference to a scalar
pub enum VxScalar {}
pub type vx_scalar = *mut VxScalar;

/// Opaque reference to a convolution
pub enum VxConvolution {}
pub type vx_convolution = *mut VxConvolution;

/// Opaque reference to a matrix
pub enum VxMatrix {}
pub type vx_matrix = *mut VxMatrix;

/// Opaque reference to a LUT
pub enum VxLUT {}
pub type vx_lut = *mut VxLUT;

/// Opaque reference to a threshold
pub enum VxThreshold {}
pub type vx_threshold = *mut VxThreshold;

/// Opaque reference to a pyramid
pub enum VxPyramid {}
pub type vx_pyramid = *mut VxPyramid;

/// Opaque reference to an image
pub enum VxImage {}
pub type vx_image = *mut VxImage;

/// Opaque reference to any reference type
pub enum VxReference {}
pub type vx_reference = *mut VxReference;

/// Node completion callback type
pub type vx_nodecomplete_f = Option<extern "C" fn(vx_node) -> vx_action>;

/// Action return values from callbacks
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub enum vx_action {
    VX_ACTION_CONTINUE = 0,
    VX_ACTION_ABANDON = 1,
}

// ============================================================================
// Internal Structures (Send + Sync)
// ============================================================================

/// Internal graph data (stored in Arc)
struct GraphData {
    id: u64,
    context_id: u32,
    nodes: Mutex<Vec<u64>>, // Store node IDs instead of pointers
}

/// Internal node data (stored in Arc)
struct NodeData {
    id: u64,
    context_id: u32,
    graph_id: u64,
    kernel_id: u64,
    parameters: Mutex<Vec<Option<u64>>>, // Store reference IDs
    callback: Mutex<Option<vx_nodecomplete_f>>,
    status: std::sync::atomic::AtomicI32,
    ref_count: std::sync::atomic::AtomicUsize,
}

/// Internal kernel data (stored in Arc)
struct KernelData {
    id: u64,
    context_id: u32,
    name: String,
    kernel_enum: i32,
    num_params: u32,
    ref_count: std::sync::atomic::AtomicUsize,
}

/// Internal parameter data (stored in Arc)
struct ParameterData {
    id: u64,
    context_id: u32,
    kernel_id: u64,
    index: u32,
    direction: vx_enum,
    data_type: vx_enum,
    state: vx_enum,
    value: Mutex<Option<u64>>, // Store reference ID
    ref_count: std::sync::atomic::AtomicUsize,
}

// ============================================================================
// Global Storage for Object Management
// ============================================================================

use once_cell::sync::Lazy;

static CONTEXTS: Lazy<Mutex<Vec<u64>>> = Lazy::new(|| {
    Mutex::new(Vec::new())
});

static GRAPHS: Lazy<Mutex<std::collections::HashMap<u64, Arc<GraphData>>>> = Lazy::new(|| {
    Mutex::new(std::collections::HashMap::new())
});

static NODES: Lazy<Mutex<std::collections::HashMap<u64, Arc<NodeData>>>> = Lazy::new(|| {
    Mutex::new(std::collections::HashMap::new())
});

static KERNELS: Lazy<Mutex<std::collections::HashMap<u64, Arc<KernelData>>>> = Lazy::new(|| {
    Mutex::new(std::collections::HashMap::new())
});

static PARAMETERS: Lazy<Mutex<std::collections::HashMap<u64, Arc<ParameterData>>>> = Lazy::new(|| {
    Mutex::new(std::collections::HashMap::new())
});

static NEXT_ID: Lazy<std::sync::atomic::AtomicU64> = Lazy::new(|| {
    std::sync::atomic::AtomicU64::new(1)
});

fn generate_id() -> u64 {
    NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst)
}

// ============================================================================
// Context Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxCreateContext() -> vx_context {
    let id = generate_id();
    let ptr = id as *mut VxContext;
    if let Ok(mut contexts) = CONTEXTS.lock() {
        contexts.push(id);
    }
    // Also register in unified registry so vxQueryReference can find it
    if let Ok(mut unified_ctxs) = UNIFIED_CONTEXTS.lock() {
        unified_ctxs.insert(ptr as usize, Arc::new(VxCContext {
            id,
            ref_count: std::sync::atomic::AtomicUsize::new(1),
        }));
    }
    ptr
}

#[no_mangle]
pub extern "C" fn vxReleaseContext(context: *mut vx_context) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let ctx = *context;
        if ctx.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = ctx as u64;
        if let Ok(mut contexts) = CONTEXTS.lock() {
            contexts.retain(|&c| c != id);
        }
        // Also remove from unified registry
        if let Ok(mut unified_ctxs) = UNIFIED_CONTEXTS.lock() {
            unified_ctxs.remove(&(ctx as usize));
        }
        *context = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Reference Management Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxRetainReference(_ref_: vx_reference) -> vx_status {
    if _ref_.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    // In this simplified implementation, we track ref counts per object type
    // For now, just return success
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxGetStatus(ref_: vx_reference) -> vx_status {
    if ref_.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    // Check if the reference is valid - in this implementation, just return success
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxGetContext(ref_: vx_reference) -> vx_context {
    if ref_.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let id = ref_ as u64;
        // Check if it's a node
        if let Ok(nodes) = NODES.lock() {
            if let Some(node) = nodes.get(&id) {
                return node.context_id as *mut VxContext;
            }
        }
        // Check if it's a graph
        if let Ok(graphs) = GRAPHS.lock() {
            for (graph_id, graph) in graphs.iter() {
                if *graph_id == id {
                    return graph.context_id as *mut VxContext;
                }
            }
        }
        // Check if it's a kernel
        if let Ok(kernels) = KERNELS.lock() {
            for (kernel_id, kernel) in kernels.iter() {
                if *kernel_id == id {
                    return kernel.context_id as *mut VxContext;
                }
            }
        }
    }
    // Default: return first context or null
    if let Ok(contexts) = CONTEXTS.lock() {
        if let Some(&first) = contexts.first() {
            return first as *mut VxContext;
        }
    }
    std::ptr::null_mut()
}

// ============================================================================
// Graph Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxCreateGraph(context: vx_context) -> vx_graph {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    let context_id = context as u32;
    let id = generate_id();
    let graph = Arc::new(GraphData {
        id,
        context_id,
        nodes: Mutex::new(Vec::new()),
    });
    
    if let Ok(mut graphs) = GRAPHS.lock() {
        graphs.insert(id, graph);
    }
    id as *mut VxGraph
}

#[no_mangle]
pub extern "C" fn vxReleaseGraph(graph: *mut vx_graph) -> vx_status {
    if graph.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let g = *graph;
        if g.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = g as u64;
        if let Ok(mut graphs) = GRAPHS.lock() {
            graphs.remove(&id);
        }
        *graph = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Node Management Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxQueryNode(
    node: vx_node,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if node.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let id = node as u64;
        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                match attribute {
                    VX_NODE_STATUS => { // VX_NODE_STATUS
                        if size >= 4 {
                            let status = node_data.status.load(std::sync::atomic::Ordering::SeqCst);
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &status as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    VX_ERROR_NOT_IMPLEMENTED
}

#[no_mangle]
pub extern "C" fn vxSetNodeAttribute(
    node: vx_node,
    _attribute: vx_enum,
    _ptr: *const c_void,
    _size: vx_size,
) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    // For now, we just validate the parameters
    if _ptr.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    // TODO: Implement actual attribute setting
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxReleaseNode(node: *mut vx_node) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let n = *node;
        if n.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = n as u64;
        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                let count = node_data.ref_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                if count <= 1 {
                    drop(nodes);
                    if let Ok(mut nodes_mut) = NODES.lock() {
                        nodes_mut.remove(&id);
                    }
                }
            }
        }
        *node = std::ptr::null_mut();
    }
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxRemoveNode(node: *mut vx_node) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let n = *node;
        if n.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = n as u64;
        
        // Remove from graph's node list
        if let Ok(nodes) = NODES.lock() {
            if let Some(node_data) = nodes.get(&id) {
                let graph_id = node_data.graph_id;
                drop(nodes);
                if let Ok(graphs) = GRAPHS.lock() {
                    if let Some(graph) = graphs.get(&graph_id) {
                        if let Ok(mut graph_nodes) = graph.nodes.lock() {
                            graph_nodes.retain(|&nid| nid != id);
                        }
                    }
                }
            }
        }
        
        // Release the node
        vxReleaseNode(node)
    }
}

#[no_mangle]
pub extern "C" fn vxAssignNodeCallback(
    node: vx_node,
    callback: vx_nodecomplete_f,
) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    let id = node as u64;
    if let Ok(nodes) = NODES.lock() {
        if let Some(node_data) = nodes.get(&id) {
            if let Ok(mut cb) = node_data.callback.lock() {
                *cb = Some(callback);
                return VX_SUCCESS;
            }
        }
    }
    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxCreateGenericNode(graph: vx_graph, kernel: vx_kernel) -> vx_node {
    if graph.is_null() || kernel.is_null() {
        return std::ptr::null_mut();
    }
    let graph_id = graph as u64;
    let kernel_id = kernel as u64;
    
    // Get kernel num_params
    let num_params = if let Ok(kernels) = KERNELS.lock() {
        if let Some(k) = kernels.get(&kernel_id) {
            k.num_params
        } else {
            4 // Default
        }
    } else {
        4 // Default
    };
    
    // Get context_id from graph
    let context_id = if let Ok(graphs) = GRAPHS.lock() {
        if let Some(g) = graphs.get(&graph_id) {
            g.context_id
        } else {
            return std::ptr::null_mut();
        }
    } else {
        return std::ptr::null_mut();
    };
    
    let id = generate_id();
    let node = Arc::new(NodeData {
        id,
        context_id,
        graph_id,
        kernel_id,
        parameters: Mutex::new(vec![None; num_params as usize]),
        callback: Mutex::new(None),
        status: std::sync::atomic::AtomicI32::new(0),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    });
    
    if let Ok(mut nodes) = NODES.lock() {
        nodes.insert(id, node);
    }
    
    // Add to graph
    if let Ok(graphs) = GRAPHS.lock() {
        if let Some(g) = graphs.get(&graph_id) {
            if let Ok(mut graph_nodes) = g.nodes.lock() {
                graph_nodes.push(id);
            }
        }
    }
    
    id as *mut VxNode
}

// ============================================================================
// Kernel Loading Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxLoadKernels(context: vx_context, module: *const vx_char) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if module.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    
    unsafe {
        let module_name = match CStr::from_ptr(module).to_str() {
            Ok(s) => s,
            Err(_) => return VX_ERROR_INVALID_PARAMETERS,
        };
        
        // Parse module name and register kernels
        // For now, we register built-in kernels based on module name
        if module_name == "openvx-core" || module_name == "openvx-vision" || module_name.is_empty() {
            // Register built-in kernels - this is a placeholder
            VX_SUCCESS
        } else {
            VX_ERROR_INVALID_PARAMETERS
        }
    }
}

#[no_mangle]
pub extern "C" fn vxUnloadKernels(context: vx_context, _module: *const vx_char) -> vx_status {
    if context.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if _module.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    // TODO: Unregister kernels from the module
    VX_SUCCESS
}

#[no_mangle]
pub extern "C" fn vxGetKernelByName(context: vx_context, name: *const vx_char) -> vx_kernel {
    if context.is_null() || name.is_null() {
        return std::ptr::null_mut();
    }
    unsafe {
        let kernel_name = match CStr::from_ptr(name).to_str() {
            Ok(s) => s,
            Err(_) => return std::ptr::null_mut(),
        };
        
        let context_id = context as u32;
        
        // Look up kernel by name
        if let Ok(kernels) = KERNELS.lock() {
            for (id, kernel) in kernels.iter() {
                if kernel.name == kernel_name && kernel.context_id == context_id {
                    kernel.ref_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    return *id as *mut VxKernel;
                }
            }
        }
        
        // Kernel not found - create it
        let id = generate_id();
        let kernel = Arc::new(KernelData {
            id,
            context_id,
            name: kernel_name.to_string(),
            kernel_enum: 0,
            num_params: 4,
            ref_count: std::sync::atomic::AtomicUsize::new(1),
        });
        
        if let Ok(mut kernels) = KERNELS.lock() {
            kernels.insert(id, kernel);
        }
        
        id as *mut VxKernel
    }
}

#[no_mangle]
pub extern "C" fn vxGetKernelByEnum(context: vx_context, kernel_e: vx_enum) -> vx_kernel {
    if context.is_null() {
        return std::ptr::null_mut();
    }
    let context_id = context as u32;
    
    // Look up kernel by enum
    if let Ok(kernels) = KERNELS.lock() {
        for (id, kernel) in kernels.iter() {
            if kernel.kernel_enum == kernel_e && kernel.context_id == context_id {
                kernel.ref_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                return *id as *mut VxKernel;
            }
        }
    }
    
    // Kernel not found - create it
    let id = generate_id();
    let kernel_name = format!("kernel_{}", kernel_e);
    let kernel = Arc::new(KernelData {
        id,
        context_id,
        name: kernel_name,
        kernel_enum: kernel_e,
        num_params: 4,
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    });
    
    if let Ok(mut kernels) = KERNELS.lock() {
        kernels.insert(id, kernel);
    }
    
    id as *mut VxKernel
}

#[no_mangle]
pub extern "C" fn vxQueryKernel(
    kernel: vx_kernel,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if kernel.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let id = kernel as u64;
        if let Ok(kernels) = KERNELS.lock() {
            if let Some(kernel_data) = kernels.get(&id) {
                match attribute {
                    VX_KERNEL_PARAMETERS => { // VX_KERNEL_PARAMETERS
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &kernel_data.num_params as *const u32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_KERNEL_NAME => { // VX_KERNEL_NAME
                        let name_bytes = kernel_data.name.as_bytes();
                        let len = name_bytes.len().min(size);
                        let ptr_u8 = ptr as *mut u8;
                        std::ptr::copy_nonoverlapping(name_bytes.as_ptr(), ptr_u8, len);
                        return VX_SUCCESS;
                    }
                    VX_KERNEL_ENUM => { // VX_KERNEL_ENUM
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &kernel_data.kernel_enum as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    VX_ERROR_NOT_IMPLEMENTED
}

#[no_mangle]
pub extern "C" fn vxGetKernelParameterByIndex(kernel: vx_kernel, index: vx_uint32) -> vx_parameter {
    if kernel.is_null() {
        return std::ptr::null_mut();
    }
    let kernel_id = kernel as u64;
    
    // Get context_id from kernel
    let context_id = if let Ok(kernels) = KERNELS.lock() {
        if let Some(k) = kernels.get(&kernel_id) {
            k.context_id
        } else {
            return std::ptr::null_mut();
        }
    } else {
        return std::ptr::null_mut();
    };
    
    let id = generate_id();
    let param = Arc::new(ParameterData {
        id,
        context_id,
        kernel_id,
        index,
        direction: 0, // VX_INPUT
        data_type: 0,
        state: 1, // VX_PARAMETER_STATE_REQUIRED
        value: Mutex::new(None),
        ref_count: std::sync::atomic::AtomicUsize::new(1),
    });
    
    if let Ok(mut params) = PARAMETERS.lock() {
        params.insert(id, param);
    }
    
    id as *mut VxParameter
}

#[no_mangle]
pub extern "C" fn vxReleaseKernel(kernel: *mut vx_kernel) -> vx_status {
    if kernel.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let k = *kernel;
        if k.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = k as u64;
        if let Ok(kernels) = KERNELS.lock() {
            if let Some(kernel_data) = kernels.get(&id) {
                let count = kernel_data.ref_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                if count <= 1 {
                    drop(kernels);
                    if let Ok(mut kernels_mut) = KERNELS.lock() {
                        kernels_mut.remove(&id);
                    }
                }
            }
        }
        *kernel = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Parameter API Functions
// ============================================================================

#[no_mangle]
pub extern "C" fn vxQueryParameter(
    param: vx_parameter,
    attribute: vx_enum,
    ptr: *mut c_void,
    size: vx_size,
) -> vx_status {
    if param.is_null() || ptr.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let id = param as u64;
        if let Ok(params) = PARAMETERS.lock() {
            if let Some(param_data) = params.get(&id) {
                match attribute {
                    VX_PARAMETER_INDEX => { // VX_PARAMETER_INDEX
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.index as *const u32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_DIRECTION => { // VX_PARAMETER_DIRECTION
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.direction as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    VX_PARAMETER_TYPE => { // VX_PARAMETER_TYPE
                        if size >= 4 {
                            let ptr_u8 = ptr as *mut u8;
                            std::ptr::copy_nonoverlapping(
                                &param_data.data_type as *const i32 as *const u8,
                                ptr_u8,
                                4,
                            );
                            return VX_SUCCESS;
                        }
                    }
                    _ => {}
                }
            }
        }
    }
    VX_ERROR_NOT_IMPLEMENTED
}

#[no_mangle]
pub extern "C" fn vxSetParameterByIndex(
    node: vx_node,
    index: vx_uint32,
    value: vx_reference,
) -> vx_status {
    if node.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if value.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let id = node as u64;
    if let Ok(nodes) = NODES.lock() {
        if let Some(node_data) = nodes.get(&id) {
            if let Ok(mut params) = node_data.parameters.lock() {
                if (index as usize) < params.len() {
                    params[index as usize] = Some(value as u64);
                    return VX_SUCCESS;
                } else {
                    return VX_ERROR_INVALID_PARAMETERS;
                }
            }
        }
    }
    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxSetParameterByReference(
    param: vx_parameter,
    value: vx_reference,
) -> vx_status {
    if param.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    if value.is_null() {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    let id = param as u64;
    if let Ok(params) = PARAMETERS.lock() {
        if let Some(param_data) = params.get(&id) {
            if let Ok(mut val) = param_data.value.lock() {
                *val = Some(value as u64);
                return VX_SUCCESS;
            }
        }
    }
    VX_ERROR_INVALID_REFERENCE
}

#[no_mangle]
pub extern "C" fn vxReleaseParameter(param: *mut vx_parameter) -> vx_status {
    if param.is_null() {
        return VX_ERROR_INVALID_REFERENCE;
    }
    unsafe {
        let p = *param;
        if p.is_null() {
            return VX_ERROR_INVALID_REFERENCE;
        }
        let id = p as u64;
        if let Ok(params) = PARAMETERS.lock() {
            if let Some(param_data) = params.get(&id) {
                let count = param_data.ref_count.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                if count <= 1 {
                    drop(params);
                    if let Ok(mut params_mut) = PARAMETERS.lock() {
                        params_mut.remove(&id);
                    }
                }
            }
        }
        *param = std::ptr::null_mut();
    }
    VX_SUCCESS
}

// ============================================================================
// Status Constants
// ============================================================================

pub const VX_SUCCESS: vx_status = 0;
pub const VX_ERROR_INVALID_REFERENCE: vx_status = -1;
pub const VX_ERROR_INVALID_PARAMETERS: vx_status = -2;
pub const VX_ERROR_INVALID_GRAPH: vx_status = -3;
pub const VX_ERROR_INVALID_NODE: vx_status = -4;
pub const VX_ERROR_INVALID_KERNEL: vx_status = -5;
pub const VX_ERROR_INVALID_CONTEXT: vx_status = -6;
pub const VX_ERROR_NOT_IMPLEMENTED: vx_status = -30;

// ============================================================================
// Type Constants
// ============================================================================

pub const VX_TYPE_INVALID: vx_enum = 0x000;
pub const VX_TYPE_CONTEXT: vx_enum = 0x801;
pub const VX_TYPE_GRAPH: vx_enum = 0x802;
pub const VX_TYPE_NODE: vx_enum = 0x803;
pub const VX_TYPE_KERNEL: vx_enum = 0x804;
pub const VX_TYPE_PARAMETER: vx_enum = 0x805;
pub const VX_TYPE_SCALAR: vx_enum = 0x80D;
pub const VX_TYPE_IMAGE: vx_enum = 0x80F;
pub const VX_TYPE_ARRAY: vx_enum = 0x80E;
pub const VX_TYPE_MATRIX: vx_enum = 0x80B;
pub const VX_TYPE_CONVOLUTION: vx_enum = 0x80C;
pub const VX_TYPE_THRESHOLD: vx_enum = 0x80A;
pub const VX_TYPE_PYRAMID: vx_enum = 0x809;
pub const VX_TYPE_LUT: vx_enum = 0x807;

// Additional data types
pub const VX_TYPE_UINT8: vx_enum = 0x003;
pub const VX_TYPE_INT8: vx_enum = 0x002;
pub const VX_TYPE_UINT16: vx_enum = 0x005;
pub const VX_TYPE_INT16: vx_enum = 0x004;
pub const VX_TYPE_UINT32: vx_enum = 0x007;
pub const VX_TYPE_INT32: vx_enum = 0x006;
pub const VX_TYPE_UINT64: vx_enum = 0x009;
pub const VX_TYPE_INT64: vx_enum = 0x008;
pub const VX_TYPE_FLOAT32: vx_enum = 0x00A;
pub const VX_TYPE_FLOAT64: vx_enum = 0x00B;
pub const VX_TYPE_BOOL: vx_enum = 0x010;

// ============================================================================
// Attribute Constants
// ============================================================================

pub const VX_NODE_STATUS: vx_enum = 0x00;
pub const VX_NODE_PARAMETERS: vx_enum = 0x05;

pub const VX_KERNEL_PARAMETERS: vx_enum = 0x00;
pub const VX_KERNEL_NAME: vx_enum = 0x01;
pub const VX_KERNEL_ENUM: vx_enum = 0x02;

pub const VX_PARAMETER_INDEX: vx_enum = 0x00;
pub const VX_PARAMETER_DIRECTION: vx_enum = 0x01;
pub const VX_PARAMETER_TYPE: vx_enum = 0x02;

// Scalar attributes
pub const VX_SCALAR_TYPE: vx_enum = 0x00;

// Threshold attributes
pub const VX_THRESHOLD_VALUE: vx_enum = 0x00;
pub const VX_THRESHOLD_LOWER: vx_enum = 0x01;
pub const VX_THRESHOLD_UPPER: vx_enum = 0x02;
pub const VX_THRESHOLD_TRUE_VALUE: vx_enum = 0x03;
pub const VX_THRESHOLD_FALSE_VALUE: vx_enum = 0x04;

// ============================================================================
// Direction Constants
// ============================================================================

pub const VX_INPUT: vx_enum = 0;
pub const VX_OUTPUT: vx_enum = 1;

// ============================================================================
// Memory and Copy Constants
// ============================================================================

pub const VX_READ_ONLY: vx_enum = 0;
pub const VX_WRITE_ONLY: vx_enum = 1;
pub const VX_READ_AND_WRITE: vx_enum = 2;
pub const VX_MEMORY_TYPE_HOST: vx_enum = 0;
pub const VX_MEMORY_TYPE_NONE: vx_enum = 1;

// ============================================================================
// Image Format Constants
// ============================================================================

pub const VX_DF_IMAGE_U8: vx_df_image = 0x000008u32;
pub const VX_DF_IMAGE_S8: vx_df_image = 0x800008u32;
pub const VX_DF_IMAGE_U16: vx_df_image = 0x000010u32;
pub const VX_DF_IMAGE_S16: vx_df_image = 0x800010u32;
pub const VX_DF_IMAGE_U32: vx_df_image = 0x000020u32;
pub const VX_DF_IMAGE_S32: vx_df_image = 0x800020u32;
pub const VX_DF_IMAGE_RGB: vx_df_image = 0x52474218;
pub const VX_DF_IMAGE_RGBX: vx_df_image = 0x58424720;
pub const VX_DF_IMAGE_RGBA: vx_df_image = 0x41424720;
pub const VX_DF_IMAGE_NV12: vx_df_image = 0x3231564C;
pub const VX_DF_IMAGE_NV21: vx_df_image = 0x3132564C;
pub const VX_DF_IMAGE_UYVY: vx_df_image = 0x59555659;
pub const VX_DF_IMAGE_YUYV: vx_df_image = 0x56595559;
pub const VX_DF_IMAGE_IYUV: vx_df_image = 0x56555949;
pub const VX_DF_IMAGE_YUV4: vx_df_image = 0x34555659;
pub const VX_DF_IMAGE_GRAYSCALE: vx_df_image = 0x0085859;

// Virtual image format
pub const VX_DF_IMAGE_VIRT: vx_df_image = 0xFFFFFFFFu32;

// ============================================================================
// Additional Status Codes
// ============================================================================

pub const VX_ERROR_INVALID_FORMAT: vx_status = -7;
pub const VX_ERROR_NO_MEMORY: vx_status = -8;

// ============================================================================
// Image Attributes
// ============================================================================

pub const VX_IMAGE_FORMAT: vx_enum = 0x00;
pub const VX_IMAGE_WIDTH: vx_enum = 0x01;
pub const VX_IMAGE_HEIGHT: vx_enum = 0x02;
pub const VX_IMAGE_PLANES: vx_enum = 0x03;

// ============================================================================
// Array Attributes
// ============================================================================

pub const VX_ARRAY_CAPACITY: vx_enum = 0x00;
pub const VX_ARRAY_ITEMTYPE: vx_enum = 0x01;
pub const VX_ARRAY_NUMITEMS: vx_enum = 0x02;
pub const VX_ARRAY_ITEMSIZE: vx_enum = 0x03;

// ============================================================================
// Opaque Types for Arrays and Structures
// ============================================================================

pub enum VxArray {}
pub type vx_array = *mut VxArray;

/// Rectangle structure
#[repr(C)]
pub struct vx_rectangle_t {
    pub start_x: vx_uint32,
    pub start_y: vx_uint32,
    pub end_x: vx_uint32,
    pub end_y: vx_uint32,
}

/// Image patch addressing structure
#[repr(C)]
pub struct vx_imagepatch_addressing_t {
    pub dim_x: vx_uint32,
    pub dim_y: vx_uint32,
    pub stride_x: vx_int32,
    pub stride_y: vx_int32,
    pub scale_x: vx_uint32,
    pub scale_y: vx_uint32,
    pub step_x: vx_uint32,
    pub step_y: vx_uint32,
}
