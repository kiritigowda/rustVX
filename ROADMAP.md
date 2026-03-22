# rustVX Development Roadmap

## Current Status (March 2026)

**Proof of Concept Complete**
- ✅ Core framework (context, graph, references)
- ✅ C API bindings (32 functions)
- ✅ Data objects (Image, Array, Scalar, etc.)
- ✅ Vision kernels with real algorithms
- ⚠️ ~10% API coverage (~32/300 functions)

## Phase 1: Complete Core API (Weeks 1-3)

### Data Object APIs
- [ ] vxCreateImage, vxCreateVirtualImage
- [ ] vxQueryImage, vxSetImageAttribute
- [ ] vxMapImagePatch, vxUnmapImagePatch
- [ ] vxCreateImageFromHandle
- [ ] vxCreateScalar, vxQueryScalar
- [ ] vxCreateArray, vxAddArrayItems, vxTruncateArray
- [ ] vxCreateConvolution, vxCopyConvolutionCoefficients
- [ ] vxCreateMatrix, vxCopyMatrix
- [ ] vxCreateLUT, vxCopyLUT
- [ ] vxCreateThreshold, vxSetThresholdAttribute
- [ ] vxCreateDistribution
- [ ] vxCreateRemap
- [ ] vxCreateObjectArray, vxGetObjectArrayItem
- [ ] vxCreatePyramid, vxGetPyramidLevel

### Graph/Node APIs
- [ ] vxQueryNode, vxSetNodeAttribute
- [ ] vxReleaseNode, vxRemoveNode
- [ ] vxAssignNodeCallback
- [ ] vxCreateGenericNode

### Kernel APIs
- [ ] vxLoadKernels, vxUnloadKernels
- [ ] vxGetKernelByName, vxGetKernelByEnum
- [ ] vxQueryKernel
- [ ] vxGetKernelParameterByIndex

### Reference Management
- [ ] vxRetainReference
- [ ] vxGetStatus
- [ ] vxGetContext

**Target:** 150+ API functions (50% coverage)

## Phase 2: Vision Algorithm Optimization (Weeks 4-6)

### SIMD Optimizations
- [ ] SSE2/AVX2 for x86_64
- [ ] NEON for ARM/AArch64
- [ ] Parallel image processing with rayon
- [ ] Separable filter optimization

### Algorithm Enhancements
- [ ] Optimized Harris corners (structure tensor)
- [ ] FAST9/12 with high-speed rejection
- [ ] Integral image with O(1) rectangle queries
- [ ] Mean-shift tracking
- [ ] Hough line detection

### Border Handling
- [ ] Complete all border modes
- [ ] Edge case optimization

**Target:** Production-ready performance

## Phase 3: CTS Conformance (Weeks 7-9)

### Integration
- [ ] Obtain official Khronos headers
- [ ] Complete CTS build
- [ ] Fix link errors

### Testing
- [ ] Run Base Feature tests
- [ ] Run Vision Conformance tests
- [ ] Debug and fix failures
- [ ] Achieve passing conformance

**Target:** Khronos Certified Vision Conformance

## Phase 4: Extensions (Weeks 10+)

### Optional Feature Sets
- [ ] Enhanced Vision Feature Set
- [ ] Binary Image Feature Set
- [ ] Neural Network extension (vx_nn)

### Platform Support
- [ ] GPU acceleration (CUDA/OpenCL/Vulkan)
- [ ] Embedded platforms (ARM Cortex, RISC-V)

## Success Criteria

1. **API Completeness:** 100% of OpenVX 1.3.1 C API
2. **Vision Conformance:** Pass Khronos CTS
3. **Performance:** Within 20% of reference implementations
4. **Memory Safety:** No unsafe code paths (Rust guarantee)

## Contributing

See CONTRIBUTING.md for how to help implement specific APIs or algorithms.
