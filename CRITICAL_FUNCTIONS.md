# Critical Functions for Conformance

## Priority 1: Data Objects (Must Have)

### Image Operations
```c
vx_image vxCreateImage(vx_context context, vx_uint32 width, vx_uint32 height, vx_df_image color);
vx_image vxCreateVirtualImage(vx_graph graph, vx_uint32 width, vx_uint32 height, vx_df_image color);
vx_status vxQueryImage(vx_image image, vx_enum attribute, void *ptr, vx_size size);
vx_status vxSetImageAttribute(vx_image image, vx_enum attribute, const void *ptr, vx_size size);
vx_status vxMapImagePatch(vx_image image, const vx_rectangle_t *rect, vx_uint32 plane_index, vx_map_id *map_id, vx_imagepatch_addressing_t *addr, void **ptr, vx_enum usage, vx_enum mem_type, vx_uint32 flags);
vx_status vxUnmapImagePatch(vx_image image, vx_map_id map_id);
```

### Arrays
```c
vx_array vxCreateArray(vx_context context, vx_enum item_type, vx_size capacity);
vx_status vxAddArrayItems(vx_array arr, vx_size count, const void *ptr, vx_stride stride);
vx_status vxTruncateArray(vx_array arr, vx_size new_num_items);
vx_status vxQueryArray(vx_array arr, vx_enum attribute, void *ptr, vx_size size);
```

### Scalars
```c
vx_scalar vxCreateScalar(vx_context context, vx_enum data_type, const void *ptr);
vx_status vxQueryScalar(vx_scalar scalar, vx_enum attribute, void *ptr, vx_size size);
```

### Convolution
```c
vx_convolution vxCreateConvolution(vx_context context, vx_size columns, vx_size rows);
vx_status vxCopyConvolutionCoefficients(vx_convolution conv, void *user_ptr, vx_enum usage, vx_enum user_mem_type);
```

### Matrix
```c
vx_matrix vxCreateMatrix(vx_context context, vx_enum data_type, vx_size columns, vx_size rows);
vx_status vxCopyMatrix(vx_matrix matrix, void *user_ptr, vx_enum usage, vx_enum user_mem_type);
```

### LUT
```c
vx_lut vxCreateLUT(vx_context context, vx_enum data_type, vx_size count);
vx_status vxCopyLUT(vx_lut lut, void *user_ptr, vx_enum usage, vx_enum user_mem_type);
```

### Threshold
```c
vx_threshold vxCreateThreshold(vx_context context, vx_enum thresh_type, vx_enum data_type);
vx_status vxSetThresholdAttribute(vx_threshold thresh, vx_enum attribute, const void *ptr, vx_size size);
```

## Priority 2: Graph Management

### Node Operations
```c
vx_status vxQueryNode(vx_node node, vx_enum attribute, void *ptr, vx_size size);
vx_status vxSetNodeAttribute(vx_node node, vx_enum attribute, const void *ptr, vx_size size);
vx_status vxReleaseNode(vx_node *node);
vx_status vxRemoveNode(vx_node *node);
vx_status vxAssignNodeCallback(vx_node node, vx_nodecomplete_f callback);
```

## Priority 3: Kernel Loading

```c
vx_status vxLoadKernels(vx_context context, const vx_char *module);
vx_status vxUnloadKernels(vx_context context, const vx_char *module);
vx_kernel vxGetKernelByName(vx_context context, const vx_char *name);
vx_kernel vxGetKernelByEnum(vx_context context, vx_enum kernel);
vx_status vxQueryKernel(vx_kernel kernel, vx_enum attribute, void *ptr, vx_size size);
```

## Priority 4: Reference Management

```c
vx_status vxRetainReference(vx_reference ref);
vx_status vxGetStatus(vx_reference ref);
vx_context vxGetContext(vx_reference ref);
```

## Implementation Notes

### Thread Safety
All data object operations must be thread-safe when accessing different objects.

### Memory Management
- Use Arc<Mutex<T>> for shared mutable state
- Use Arc for reference counting
- Implement proper drop handlers

### Error Handling
- Return VX_ERROR_INVALID_REFERENCE for null pointers
- Return VX_ERROR_INVALID_PARAMETERS for bad arguments
- Use Rust's Result type internally, convert to vx_status for C API

### Testing
For each function, create unit tests:
1. Happy path (valid inputs)
2. Error cases (null pointers, invalid parameters)
3. Edge cases (zero size, maximum size)
