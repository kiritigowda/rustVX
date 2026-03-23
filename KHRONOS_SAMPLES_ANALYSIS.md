# Khronos OpenVX Sample Implementation - Examples Analysis for rustVX

## Overview

This document analyzes the example code from the Khronos OpenVX Sample Implementation (`OpenVX-sample-impl/examples/`) and maps them to the rustVX implementation, identifying patterns, best practices, and potential porting strategies.

---

## 1. Basic Graph Patterns

### 1.1 Single Node Graph (`vx_single_node_graph.c`)

**C Pattern:**
```c
vx_context context = vxCreateContext();
vx_image in = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
vx_image out = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
vx_graph graph = vxCreateGraph(context);
vx_node node = vxGaussian3x3Node(graph, in, out);
status = vxVerifyGraph(graph);
if (status == VX_SUCCESS)
    status = vxProcessGraph(graph);
// VXU immediate mode alternative
status = vxuGaussian3x3(context, in, out);
```

**rustVX Equivalent:**
```rust
use openvx_core::{Context, Graph};
use openvx_image::Image;
use openvx_vision::filter;

let context = Context::new()?;
let in_img = Image::create(&context, width, height, ImageFormat::U8)?;
let out_img = Image::create(&context, width, height, ImageFormat::U8)?;
let mut graph = Graph::create(&context)?;
let node = filter::gaussian3x3_node(&mut graph, &in_img, &out_img)?;
graph.verify()?;
graph.process()?;
```

**Key Observations:**
- ✅ rustVX has equivalent `Context`, `Graph`, `Image` types
- ✅ Vision kernels are implemented in `openvx-vision`
- ⚠️ Need to verify immediate mode VXU functions exist

---

### 1.2 Multi-Node Graph with Virtual Images (`vx_multi_node_graph.c`)

**C Pattern:**
```c
vx_image blurred = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
vx_image gx = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
vx_image gy = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);

vx_node nodes[] = {
    vxGaussian3x3Node(graph, in, blurred),
    vxSobel3x3Node(graph, blurred, gx, gy),
    vxMagnitudeNode(graph, gx, gy, out),
};
```

**rustVX Equivalent:**
```rust
let blurred = Image::create_virtual(&graph, 0, 0, ImageFormat::Virt)?;
let gx = Image::create_virtual(&graph, 0, 0, ImageFormat::S16)?;
let gy = Image::create_virtual(&graph, 0, 0, ImageFormat::S16)?;

let node1 = filter::gaussian3x3_node(&mut graph, &in_img, &blurred)?;
let node2 = gradient::sobel3x3_node(&mut graph, &blurred, &gx, &gy)?;
let node3 = gradient::magnitude_node(&mut graph, &gx, &gy, &out_img)?;
```

**Key Observations:**
- Virtual images allow optimization (no actual buffer allocation)
- Graph nodes form a DAG (directed acyclic graph)
- rustVX implements these patterns in `openvx-vision`

---

### 1.3 Independent Node Execution (`vx_independent.c`)

**C Pattern:**
```c
vxChannelExtractNode(graph, images[0], VX_CHANNEL_Y, virts[0]);
vxGaussian3x3Node(graph, virts[0], virts[1]);
vxSobel3x3Node(graph, virts[1], virts[2], virts[3]);
vxMagnitudeNode(graph, virts[2], virts[3], images[1]);
vxPhaseNode(graph, virts[2], virts[3], images[2]);
```

**Key Insight:** Nodes without dependencies can execute in parallel. The sample demonstrates parallel execution of `Magnitude` and `Phase` nodes.

**rustVX Status:**
- ✅ Vision functions implemented
- ⚠️ Parallel execution in graph scheduler needs verification

---

## 2. Graph Factory Pattern (`vx_graph_factory.c`)

### 2.1 Factory Architecture

**C Pattern:**
```c
typedef struct {
    vx_enum factory_name;
    vx_graph (*factory)(vx_context);
} vx_graph_factory_t;

vx_graph_factory_t factories[] = {
    {VX_GRAPH_FACTORY_EDGE, vxEdgeGraphFactory},
    {VX_GRAPH_FACTORY_CORNERS, vxCornersGraphFactory},
    {VX_GRAPH_FACTORY_PIPELINE, vxPipelineGraphFactory},
};
```

**Key Benefits:**
1. **Encapsulation**: Graph construction logic is hidden
2. **Reusability**: Same factory creates multiple graph instances
3. **Parameterization**: Graph parameters abstract internal details
4. **Extensibility**: New factories can be added

**rustVX Implementation:**
```rust
pub trait GraphFactory {
    fn create(context: &Context) -> Result<Graph, VxError>;
}

pub struct EdgeGraphFactory;
impl GraphFactory for EdgeGraphFactory {
    fn create(context: &Context) -> Result<Graph, VxError> {
        // Implementation
    }
}
```

---

### 2.2 Edge Detection Factory (`vx_factory_edge.c`)

**C Implementation:**
```c
vx_graph vxEdgeGraphFactory(vx_context c) {
    vx_kernel kernels[] = {
        vxGetKernelByEnum(c, VX_KERNEL_GAUSSIAN_3x3),
        vxGetKernelByEnum(c, VX_KERNEL_SOBEL_3x3),
        vxGetKernelByEnum(c, VX_KERNEL_MAGNITUDE),
    };
    
    vx_image virts[] = {
        vxCreateVirtualImage(g, 0, 0, VX_DF_IMAGE_VIRT), // blurred
        vxCreateVirtualImage(g, 0, 0, VX_DF_IMAGE_VIRT), // gx
        vxCreateVirtualImage(g, 0, 0, VX_DF_IMAGE_VIRT), // gy
    };
    
    vx_node nodes[] = {
        vxCreateGenericNode(g, kernels[0]), // Gaussian
        vxCreateGenericNode(g, kernels[1]), // Sobel
        vxCreateGenericNode(g, kernels[2]), // Mag
    };
    
    // Expose only input/output as graph parameters
    vx_parameter params[] = {
        vxGetParameterByIndex(nodes[0], 0), // input
        vxGetParameterByIndex(nodes[2], 2), // output
    };
    vxAddParameterToGraph(g, params[p]);
}
```

**Pipeline:** Gaussian Blur → Sobel (Gx, Gy) → Magnitude

**rustVX Equivalent:**
```rust
pub fn edge_detection_factory(context: &Context) -> Result<Graph, VxError> {
    let mut graph = Graph::create(context)?;
    
    let blurred = Image::create_virtual(&graph, 0, 0, ImageFormat::Virt)?;
    let gx = Image::create_virtual(&graph, 0, 0, ImageFormat::Virt)?;
    let gy = Image::create_virtual(&graph, 0, 0, ImageFormat::Virt)?;
    
    let gaussian = filter::gaussian3x3_node(&mut graph, 
        &graph.get_parameter(0)?, &blurred)?;
    let sobel = gradient::sobel3x3_node(&mut graph, 
        &blurred, &gx, &gy)?;
    let magnitude = gradient::magnitude_node(&mut graph, 
        &gx, &gy, &graph.get_parameter(1)?)?;
    
    Ok(graph)
}
```

---

### 2.3 Corner Detection Factory (`vx_factory_corners.c`)

**C Implementation:**
```c
vx_graph vxCornersGraphFactory(vx_context context) {
    // Harris Corners parameters
    vx_float32 strength_thresh = 10000.0f;
    vx_float32 r = 1.5f;
    vx_float32 sensitivity = 0.14f;
    vx_int32 window_size = 3;
    vx_int32 block_size = 3;
    vx_enum channel = VX_CHANNEL_Y;
    
    // Pipeline: Channel Extract → Median Filter → Harris Corners
    vx_node nodes[] = {
        vxCreateGenericNode(graph, kernels[0]), // Channel Extract
        vxCreateGenericNode(graph, kernels[1]), // Median 3x3
        vxCreateGenericNode(graph, kernels[2]), // Harris Corners
    };
}
```

**Pipeline:** Channel Extract → Median 3x3 → Harris Corners

**Key Features:**
- Uses scalars for kernel parameters
- Pre-processing with median filter for noise reduction
- Graph parameters: input image, output corners array

---

### 2.4 Vision Pipeline Factory (`vx_factory_pipeline.c`)

**C Implementation:**
```c
vx_graph vxPipelineGraphFactory(vx_context context) {
    // Extended pipeline with callback
    vx_node nodes[] = {
        vxChannelExtractNode(graph, ...),        // Extract Y channel
        vxSobel3x3Node(graph, ...),              // Edge detection
        vxMagnitudeNode(graph, ...),             // Magnitude
        vxMinMaxLocNode(graph, ...),             // Find max
        vxThresholdNode(graph, ...),             // Apply threshold
        vxCreateGenericNode(graph, xyz_kernel),  // Custom kernel
    };
    
    // Assign callback to MinMaxLoc node
    vxAssignNodeCallback(nodes[3], example_maximacallback);
}

// Callback function
vx_action VX_CALLBACK example_maximacallback(vx_node node) {
    vx_uint32 max_intensity = 0;
    // Read max value from scalar
    if (max_intensity > 10)
        return VX_ACTION_CONTINUE;
    else
        return VX_ACTION_ABANDON;  // Skip if edges weak
}
```

**Pipeline:** Channel Extract → Sobel → Magnitude → MinMaxLoc (with callback) → Threshold → Custom XYZ

**Key Features:**
- **Node Callbacks**: Conditionally abandon graph based on runtime values
- **VX_ACTION_CONTINUE**: Proceed with graph execution
- **VX_ACTION_ABANDON**: Skip remaining nodes

**rustVX Status:**
- ⚠️ Node callbacks not yet implemented
- ⚠️ VX_ACTION semantics need implementation

---

## 3. Node Callbacks (`vx_callback.c`)

### 3.1 Callback Pattern

**C Implementation:**
```c
vx_action VX_CALLBACK analyze_brightness(vx_node node) {
    vx_action action = VX_ACTION_ABANDON;
    vx_parameter pmax = vxGetParameterByIndex(node, 2); // Max Value
    
    vx_scalar smax = 0;
    vxQueryParameter(pmax, VX_PARAMETER_REF, &smax, sizeof(smax));
    
    vx_uint8 value = 0u;
    vxCopyScalar(smax, &value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    
    if (value >= MY_DESIRED_THRESHOLD) {
        action = VX_ACTION_CONTINUE;
    }
    return action;
}

// Usage
vx_node node = vxMinMaxLocNode(graph, input, ...);
vxAssignNodeCallback(node, &analyze_brightness);
```

**Key Use Cases:**
1. Dynamic graph control based on intermediate results
2. Early termination when conditions are met
3. Conditional processing based on image statistics

**rustVX Implementation Needed:**
```rust
pub trait NodeCallback: Send + Sync {
    fn on_node_complete(&self, node: &Node) -> VxAction;
}

pub enum VxAction {
    Continue,   // Proceed with graph execution
    Abandon,    // Skip remaining nodes
}
```

---

## 4. Tiling Extension (`vx_tiling_*.c`)

### 4.1 Tiling Kernel Pattern

**C Implementation:**
```c
// Tiling function signature
void add_image_tiling(void * VX_RESTRICT parameters[VX_RESTRICT],
                      void * VX_RESTRICT tile_memory,
                      vx_size tile_memory_size)
{
    vx_tile_t *in0 = (vx_tile_t *)parameters[0];
    vx_tile_t *in1 = (vx_tile_t *)parameters[1];
    vx_tile_t *out = (vx_tile_t *)parameters[2];

    for (j = 0u; j < vxTileHeight(out,0); j+=vxTileBlockHeight(out)) {
        for (i = 0u; i < vxTileWidth(out,0); i+=vxTileBlockWidth(out)) {
            vx_uint16 pixel = vxImagePixel(vx_uint8, in0, 0, i, j, 0, 0) +
                              vxImagePixel(vx_uint8, in1, 0, i, j, 0, 0);
            if (pixel > INT16_MAX) pixel = INT16_MAX;
            vxImagePixel(vx_int16, out, 0, i, j, 0, 0) = (vx_int16)pixel;
        }
    }
}
```

**Key Features:**
- Tile-based processing for cache efficiency
- Block sizes (1x1, 16x16, flexible)
- Neighborhood handling for filters

**rustVX Status:**
- ⚠️ Tiling extension not yet implemented
- Custom kernel support needed

---

### 4.2 Tiling Main Example (`vx_tiling_main.c`)

**C Usage Pattern:**
```c
vxLoadKernels(context, "openvx-tiling");
vxLoadKernels(context, "openvx-debug");

vx_node nodes[] = {
    vxFReadImageNode(graph, "lena_512x512.pgm", images[1]),
    vxTilingBoxNode(graph, images[1], images[2], 5, 5),
    vxFWriteImageNode(graph, images[2], "ot_box_lena_512x512.pgm"),
    vxTilingGaussianNode(graph, images[1], images[3]),
    vxFWriteImageNode(graph, images[3], "ot_gauss_lena_512x512.pgm"),
};
```

**Key Kernels:**
- `vxTilingAddNode`: Element-wise addition
- `vxTilingAlphaNode`: Alpha blending
- `vxTilingBoxNode`: Box filter (MxN)
- `vxTilingGaussianNode`: Gaussian filter

**rustVX Status:**
- ⚠️ Debug extension (FRead/FWrite) not implemented
- ⚠️ Tiling extension not implemented

---

## 5. Super Resolution (`vx_super_res.c`)

### 5.1 Multi-Graph Pipeline

**C Architecture:**
```c
vx_graph graphs[] = {
    vxCreateGraph(context),  // Graph 0: Motion estimation
    vxCreateGraph(context),  // Graph 1: Optical flow
    vxCreateGraph(context),  // Graph 2: Feature detection
    vxCreateGraph(context),  // Graph 3: Final composition
};

// Graph 0: Iterative refinement
vxChannelExtractNode(graphs[0], images[0], VX_CHANNEL_Y, images[1]);
vxScaleImageNode(graphs[0], images[1], images[2], VX_INTERPOLATION_BILINEAR);
vxWarpPerspectiveNode(graphs[0], images[2], matrix_forward, 0, images[3]);
vxGaussian3x3Node(graphs[0], images[3], images[4]);
...
vxAccumulateWeightedImageNode(graphs[0], images[9], alpha_s, images[10]);

// Graph 1: Optical Flow
vxGaussianPyramidNode(graphs[1], images[1], pyramid_new);
vxOpticalFlowPyrLKNode(graphs[1], pyramid_old, pyramid_new, ...);

// Graph 2: Feature Detection
vxHarrisCornersNode(graphs[2], images[1], ...);
vxGaussianPyramidNode(graphs[2], images[1], pyramid_old);

// Graph 3: Final output
vxSubtractNode(graphs[3], ...);
vxChannelCombineNode(graphs[3], ...);
```

**Pipeline:**
1. **Graph 2 (Init)**: Harris corners + pyramid for initial frame
2. **Graph 1 (Per Frame)**: Optical flow tracking
3. **Graph 0 (Per Frame)**: Motion estimation + accumulation
4. **Graph 3 (Final)**: Combine and output

**Key Features:**
- Multi-graph coordination
- User callbacks between graphs
- Pyramid-based optical flow
- Image accumulation

**rustVX Status:**
- ✅ Harris corners, pyramids, optical flow implemented
- ⚠️ `vxWarpPerspectiveNode` - needs verification
- ⚠️ `vxAccumulateWeightedImageNode` - needs verification
- ⚠️ Multi-graph coordination patterns need examples

---

## 6. Introspection (`vx_introspection.c`)

### 6.1 Kernel Discovery Pattern

**C Implementation:**
```c
// Query number of kernels
vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNELS, &num_kernels, sizeof(num_kernels));

// Allocate and fill kernel table
vx_size size = num_kernels * sizeof(vx_kernel_info_t);
vx_kernel_info_t *table = (vx_kernel_info_t *)malloc(size);
vxQueryContext(context, VX_CONTEXT_UNIQUE_KERNEL_TABLE, table, size);

// Iterate kernels
for (k = 0; k < num_kernels; k++) {
    vx_kernel kernel = vxGetKernelByName(context, table[k].name);
    
    // Query kernel attributes
    vx_uint32 num_params = 0u;
    vxQueryKernel(kernel, VX_KERNEL_PARAMETERS, &num_params, sizeof(num_params));
    
    // Query parameters
    for (p = 0; p < num_params; p++) {
        vx_parameter param = vxGetKernelParameterByIndex(kernel, p);
        vxQueryParameter(param, VX_PARAMETER_DIRECTION, &dir, sizeof(dir));
        vxQueryParameter(param, VX_PARAMETER_STATE, &state, sizeof(state));
        vxQueryParameter(param, VX_PARAMETER_TYPE, &type, sizeof(type));
    }
}
```

**Use Cases:**
1. **Documentation Generation**: Auto-generate kernel docs
2. **Validation**: Verify kernel signatures
3. **IDE Support**: Auto-complete, type checking
4. **Debugging**: Inspect runtime kernel configuration

**rustVX Implementation:**
```rust
pub fn introspect_kernels(context: &Context) -> Result<Vec<KernelInfo>, VxError> {
    let num_kernels = context.query_unique_kernels()?;
    let mut kernels = Vec::new();
    
    for i in 0..num_kernels {
        let kernel = context.get_kernel_by_index(i)?;
        let params = kernel.query_parameters()?;
        kernels.push(KernelInfo { name: kernel.name(), params });
    }
    
    Ok(kernels)
}
```

---

## 7. Other Examples

### 7.1 Image Patch Access (`vx_imagepatch.c`)

**C Pattern:**
```c
vx_imagepatch_addressing_t addr;
void *base = NULL;
vx_rectangle_t rect = {0, 0, width, height};

vxAccessImagePatch(image, &rect, 0, &addr, &base, VX_READ_ONLY);
// Process pixels...
vxCommitImagePatch(image, &rect, 0, &addr, base);
```

**rustVX Equivalent:**
```rust
let patch = image.map_patch(Rect::new(0, 0, width, height), AccessMode::Read)?;
// Process via slice...
image.unmap_patch(patch)?;
```

### 7.2 Delay Graph (`vx_delaygraph.c`)

**C Pattern:**
```c
vx_delay delay = vxCreateDelay(context, (vx_reference)image, 3);
vxAgeDelay(delay);
```

**Key Features:**
- Temporal buffering for video processing
- Automatic age management

### 7.3 Extensions (`vx_extensions.c`)

**C Pattern:**
```c
vxLoadKernels(context, "xyz");  // Load custom kernel library
vxUnloadKernels(context, "xyz");
```

---

## 8. Coverage Analysis

### 8.1 Implemented in rustVX

| Feature | C API | rustVX Status |
|---------|-------|---------------|
| Context | ✅ | ✅ |
| Graph | ✅ | ✅ |
| Image | ✅ | ✅ |
| Virtual Images | ✅ | ✅ |
| Nodes | ✅ | ✅ |
| Graph Verify | ✅ | ✅ |
| Graph Process | ✅ | ✅ |
| VXU Immediate | ✅ | ⚠️ Needs verification |
| Gaussian Filter | ✅ | ✅ |
| Sobel Filter | ✅ | ✅ |
| Magnitude | ✅ | ✅ |
| Harris Corners | ✅ | ✅ |
| Channel Extract | ✅ | ✅ |
| Median Filter | ✅ | ✅ |
| MinMaxLoc | ✅ | ✅ |
| Threshold | ✅ | ✅ |
| Pyramids | ✅ | ✅ |
| Optical Flow | ✅ | ✅ |
| Image Add/Subtract | ✅ | ✅ |
| Scale/Remap | ✅ | ✅ |

### 8.2 Not Yet Implemented in rustVX

| Feature | Priority | Notes |
|---------|----------|-------|
| Node Callbacks | High | Required for conditional execution |
| Graph Parameters | High | Factory pattern requires this |
| Tiling Extension | Medium | Performance optimization |
| Debug Extension | Low | File I/O nodes |
| User Kernels | Medium | Custom kernel loading |
| Warp Perspective | Medium | Geometric transform |
| Image Accumulation | Medium | Super resolution use case |
| Delay Objects | Low | Video processing |

---

## 9. Ported Examples

### Example 1: Simple Edge Detection (Ported)

```rust
use openvx_core::{Context, Graph, VxError};
use openvx_image::Image;
use openvx_vision::{filter, gradient};

fn single_node_graph() -> Result<(), VxError> {
    let context = Context::new()?;
    let width = 320;
    let height = 240;
    
    let input = Image::create(&context, width, height, ImageFormat::U8)?;
    let output = Image::create(&context, width, height, ImageFormat::U8)?;
    
    let mut graph = Graph::create(&context)?;
    filter::gaussian3x3_node(&mut graph, &input, &output)?;
    
    graph.verify()?;
    graph.process()?;
    
    Ok(())
}

fn multi_node_graph() -> Result<(), VxError> {
    let context = Context::new()?;
    let width = 320;
    let height = 240;
    
    let input = Image::create(&context, width, height, ImageFormat::U8)?;
    let output = Image::create(&context, width, height, ImageFormat::U8)?;
    
    let mut graph = Graph::create(&context)?;
    
    // Virtual images for intermediate results
    let blurred = Image::create_virtual(&graph, 0, 0, ImageFormat::Virt)?;
    let gx = Image::create_virtual(&graph, 0, 0, ImageFormat::S16)?;
    let gy = Image::create_virtual(&graph, 0, 0, ImageFormat::S16)?;
    
    // Pipeline: Gaussian → Sobel → Magnitude
    filter::gaussian3x3_node(&mut graph, &input, &blurred)?;
    gradient::sobel3x3_node(&mut graph, &blurred, &gx, &gy)?;
    gradient::magnitude_node(&mut graph, &gx, &gy, &output)?;
    
    graph.verify()?;
    graph.process()?;
    
    Ok(())
}
```

### Example 2: Graph Factory Pattern (Ported)

```rust
pub struct EdgeGraphFactory;

impl EdgeGraphFactory {
    pub fn create(context: &Context) -> Result<Graph, VxError> {
        let mut graph = Graph::create(context)?;
        
        // Internal virtual images
        let blurred = Image::create_virtual(&graph, 0, 0, ImageFormat::Virt)?;
        let gx = Image::create_virtual(&graph, 0, 0, ImageFormat::S16)?;
        let gy = Image::create_virtual(&graph, 0, 0, ImageFormat::S16)?;
        
        // Pipeline
        let gaussian = filter::gaussian3x3_node(&mut graph, 
            &graph.input(0)?, &blurred)?;
        let sobel = gradient::sobel3x3_node(&mut graph, 
            &blurred, &gx, &gy)?;
        let magnitude = gradient::magnitude_node(&mut graph, 
            &gx, &gy, &graph.output(0)?)?;
        
        Ok(graph)
    }
}

// Usage
let graph = EdgeGraphFactory::create(&context)?;
graph.set_parameter(0, &input_image)?;
graph.set_parameter(1, &output_image)?;
graph.verify()?;
graph.process()?;
```

---

## 10. Recommendations

### 10.1 High Priority Additions

1. **Graph Parameters API**
   - `vxAddParameterToGraph` equivalent
   - `vxSetGraphParameterByIndex` equivalent
   - Factory pattern support

2. **Node Callbacks**
   - `vxAssignNodeCallback` equivalent
   - `VX_ACTION` enum support
   - Safe Rust callback interface

3. **VXU Immediate Mode**
   - Verify all VXU functions work standalone
   - Document usage patterns

### 10.2 Medium Priority

1. **Tiling Extension**
   - Custom kernel support
   - Tile-based processing API

2. **User Kernel Loading**
   - Dynamic kernel registration
   - Extension module system

### 10.3 Documentation

1. Create rust equivalents for all Khronos examples
2. Performance comparisons (C vs Rust)
3. Best practices guide

---

## 11. Conclusion

The rustVX implementation covers the majority of core OpenVX features demonstrated in the Khronos sample implementation. Key gaps exist in:

1. **Advanced Graph Features**: Parameters, callbacks
2. **Extensions**: Tiling, debug, user kernels
3. **Examples**: Port all Khronos examples to Rust

The architecture is well-positioned to support these additions, with clean trait-based abstractions that can accommodate the missing functionality.
