# Image Clone Strategy Implementation

## Overview

This implementation adds image cloning functionality to rustVX, enabling deep copies of OpenVX images for use in conformance testing and graph operations.

## Functions Added

### C API Functions

#### `vxCloneImage`
```c
vx_image vxCloneImage(vx_context context, vx_image source);
```
Creates a deep copy of the source image.

**Parameters:**
- `context` - The OpenVX context
- `source` - The source image to clone

**Returns:**
- A new `vx_image` handle that is a deep copy of the source, or null on error

**Behavior:**
- Validates context and source image parameters
- Queries source image dimensions and format
- Creates a new image with same properties
- Performs deep copy of pixel data
- Copies mapped patches metadata
- For virtual images without data, creates a regular image with same dimensions

#### `vxCloneImageWithGraph`
```c
vx_image vxCloneImageWithGraph(vx_context context, vx_graph graph, vx_image source);
```
Clones an image with graph awareness for CTS compatibility.

**Parameters:**
- `context` - The OpenVX context (used if source is not virtual)
- `graph` - The OpenVX graph (used if source is virtual)
- `source` - The source image to clone

**Returns:**
- A new `vx_image` handle that is a clone of the source, or null on error

**Behavior:**
- If source is virtual or has no data, creates a virtual image associated with the graph
- Otherwise, delegates to `vxCloneImage` for a regular deep copy

### Rust API Function

#### `clone_image`
```rust
pub fn clone_image(source: &Image) -> Image
```
Creates a deep copy of a Rust Image struct.

**Parameters:**
- `source` - Reference to the source Image

**Returns:**
- A new Image with cloned pixel data

## Implementation Details

### File Changes

1. **`openvx-image/src/c_api.rs`**
   - Added `vxCloneImage` function (lines 1147-1235)
   - Added `vxCloneImageWithGraph` function (lines 1237-1286)
   - Handles deep copying of image data with proper error handling

2. **`openvx-image/src/lib.rs`**
   - Exported `vxCloneImage` and `vxCloneImageWithGraph` in the C API
   - Added `clone_image` function to the Rust API (lines 108-120)

3. **`openvx-core/src/unified_c_api.rs`**
   - Re-exported clone functions for unified access (lines 1542-1547)

### Clone Strategy

1. **Parameter Validation**
   - Null checks for context and source image
   - Returns null if validation fails

2. **Property Extraction**
   - Query source image width, height, and format
   - Determine if source is virtual (no allocated data)

3. **Destination Creation**
   - For regular images: Create new image with `vxCreateImage`
   - For virtual images: Create regular image (virtual → concrete conversion)

4. **Data Copy**
   - Lock source data for reading
   - Lock destination data for writing
   - Perform `copy_from_slice` for deep copy
   - Handle size mismatches (should not occur with correct `vxCreateImage`)

5. **Metadata Copy**
   - Clone mapped patches metadata from source to destination
   - This preserves any mapping state for consistency

6. **Error Handling**
   - On any error, release the partially created destination image
   - Return null to indicate failure

### CTS Compatibility

The implementation is designed to be compatible with the OpenVX CTS test utilities, specifically the `ct_clone_image_impl` function found in `openvx-cts/test_engine/test_utils.c`. The CTS uses cloning for:

- Creating reference images for comparison
- Copying virtual images to concrete images
- Test isolation (preventing modifications to input data)

## Testing

The implementation includes error handling for:
- Null context
- Null source image
- Failed image creation
- Data lock failures
- Size mismatches

### Example Usage

```c
// Clone a regular image
vx_context context = vxCreateContext();
vx_image source = vxCreateImage(context, 640, 480, VX_DF_IMAGE_U8);
vx_image clone = vxCloneImage(context, source);

// Clone with graph (for CTS compatibility)
vx_graph graph = vxCreateGraph(context);
vx_image virtual_img = vxCreateVirtualImage(graph, 640, 480, VX_DF_IMAGE_U8);
vx_image cloned = vxCloneImageWithGraph(context, graph, virtual_img);
```

```rust
use openvx_image::{Image, ImageFormat, clone_image};

let source = Image::new(640, 480, ImageFormat::Gray);
let clone = clone_image(&source);
```

## Build Status

✅ Compiles successfully with `cargo build`

## Future Enhancements

Potential improvements for future iterations:

1. **Multi-plane Support**: Enhanced handling for planar formats (YUV, NV12, etc.)
2. **Reference Counting**: Proper reference counting integration
3. **Attribute Copying**: Full attribute copying (space, valid rectangles)
4. **Performance Optimization**: Direct memory mapping for large images
5. **Testing**: Unit tests for various image formats
