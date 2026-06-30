# rustVX OpenVX Conformance Coverage

This document maps the rustVX CI workflows to the upstream Khronos
[OpenVX Conformance Test Suite](https://github.com/KhronosGroup/OpenVX-cts)
feature sets. It tracks which conformance areas are currently exercised,
which are still gap work, and what the expected CI outcome is.

## Workflows

| Workflow | Purpose | Build feature flags |
|:---|:---|:---|
| `.github/workflows/conformance.yml` | Stable, focused CI gates and performance benchmarks | Vision + Enhanced Vision + User Data Object + Pipelining |
| `.github/workflows/ci.yml` | **Full conformance exploration** — turns on every Khronos feature set and extension so gaps are visible | Vision + Enhanced Vision + Neural Networks + NN/16 + NNEF Import + IX + Pipelining + Streaming + User Data Object + U1 |

## Conformance feature matrix

| OpenVX feature / extension | CTS compile flag | rustVX support | Exercised in `conformance.yml` | Exercised in `ci.yml` | Notes |
|:---|:---|:---:|:---:|:---:|:---|
| Core / Base conformance | (always) | ✅ | ✅ | ✅ | Graph, smoke, target, logging |
| Vision conformance | `OPENVX_CONFORMANCE_VISION` | ✅ | ✅ | ✅ | Core 2D vision kernels |
| Enhanced Vision | `OPENVX_USE_ENHANCED_VISION` | ✅ | ✅ | ✅ | Tensors, HOG, LBP, bilateral, control flow |
| User Data Object KHR | `OPENVX_USE_USER_DATA_OBJECT` | ✅ | ✅ | ✅ | `test_user_data_object.c` |
| Pipelining KHR | `OPENVX_USE_PIPELINING` | ✅ | ✅ | ✅ | `test_graph_pipeline.c` |
| Streaming KHR | `OPENVX_USE_STREAMING` | ⚠️ partial | ❌ | ✅ | `test_graph_streaming.c` |
| Neural Networks (NN) KHR | `OPENVX_USE_NN` / `OPENVX_CONFORMANCE_NEURAL_NETWORKS` | ⚠️ partial | ❌ | ✅ | `test_tensor_nn.c` |
| NN 16-bit extension | `OPENVX_USE_NN_16` | ⚠️ partial | ❌ | ✅ | `test_tensor_networks.c` |
| NNEF Import KHR | `OPENVX_CONFORMANCE_NNEF_IMPORT` | ⚠️ partial | ❌ | ✅ | `test_nnef_import.c` |
| Import / Export (IX) KHR | `OPENVX_USE_IX` | ⚠️ partial | ❌ | ✅ | `test_export_import_extension.c` |
| Binary (U1) feature set | `OPENVX_USE_U1` | ✅ | ✅ | ✅ | 1-bit image / tensor paths |

Legend: ✅ exercised and expected to pass; ⚠️ exercised but known gaps / partial implementation; ❌ not exercised.

## What the new `ci.yml` adds

The upstream Khronos `OpenVX-sample-impl` workflow (`.github/workflows/ci.yml`)
builds the CTS with **all** feature flags enabled and then runs the suite in
broad bands:

- baseline / smoke / coverage
- enhanced vision
- neural networks
- import/export (IX)
- graph features (delay, ROI, callbacks, pipeline, streaming)
- data objects
- user-defined kernels

rustVX's original `conformance.yml` mirrors much of that split, but it
deliberately disables several feature flags because the corresponding
rustVX implementation was not complete enough. The new `.github/workflows/ci.yml`
file turns those flags **on** so the project can track remaining conformance
work transparently. Each new band is run with `continue-on-error: true` until
the implementation is mature, after which the flag should be removed and the
job promoted to a required gate.

### Gaps surfaced by the new workflow

1. **Neural Network layers** — rustVX has tensor data-object support but does
   not yet implement the NN layer nodes (`vxConvolutionLayer`,
   `vxFullyConnectedLayer`, `vxPoolingLayer`, etc.). Building the CTS with
   `OPENVX_CONFORMANCE_NEURAL_NETWORKS=ON` will therefore compile against the
   header APIs but the runtime tests will fail until those nodes are wired up.

2. **NNEF import** — `vxImportNNEFKernel` / the NNEF import kernel path needs
   the NNEF-Tools parser integration. The CTS compiles this path only when
   `OPENVX_CONFORMANCE_NNEF_IMPORT=ON`.

3. **Import/Export (IX)** — rustVX currently exports stub implementations for
   `vxExportObjectsToMemory` / `vxImportObjectsFromMemory`. The IX tests will
   exercise these stubs and fail until real serialization is implemented.

4. **Streaming** — `test_graph_streaming.c` relies on pipelining and streaming
   graph-scheduling semantics. rustVX has pipelining but full streaming
   semantics are still partial.

## How to run locally

```bash
# Build rustVX release library (x86-64 with AVX2)
RUSTFLAGS="-C target-cpu=x86-64-v3" \
  cargo build --release -p openvx-ffi \
  --features "openvx-core/sse2 openvx-core/avx2 openvx-vision/sse2 openvx-vision/avx2"

# Build the CTS with the full feature set
cd OpenVX-cts
mkdir -p build-full && cd build-full
cmake .. \
  -DOPENVX_INCLUDES="$PWD/../../include;$PWD/../include" \
  -DOPENVX_LIBRARIES="$PWD/../../target/release/libopenvx_ffi.so;m" \
  -DOPENVX_CONFORMANCE_VISION=ON \
  -DOPENVX_USE_ENHANCED_VISION=ON \
  -DOPENVX_CONFORMANCE_NEURAL_NETWORKS=ON \
  -DOPENVX_USE_NN=ON \
  -DOPENVX_USE_NN_16=ON \
  -DOPENVX_CONFORMANCE_NNEF_IMPORT=ON \
  -DOPENVX_USE_IX=ON \
  -DOPENVX_USE_PIPELINING=ON \
  -DOPENVX_USE_STREAMING=ON \
  -DOPENVX_USE_USER_DATA_OBJECT=ON \
  -DOPENVX_USE_U1=ON
make -j$(nproc)

# Run a single band, e.g. IX
export LD_LIBRARY_PATH=$PWD/../../target/release
export VX_TEST_DATA_PATH=$PWD/../test_data/
./bin/vx_test_conformance --filter="Graph/ExportImport*:*IX*"
```

## Future work

1. Implement the NN KHR layer nodes and enable `cts-neural-networks` as a
   required (non-`continue-on-error`) job.
2. Implement real IX object serialization and remove `continue-on-error`
   from `cts-ix`.
3. Complete streaming graph semantics and remove `continue-on-error` from
   `cts-graph-features`.
4. Add NNEF-Tools parser integration so `cts-nnef-import` can run.
5. Once all bands pass, fold `ci.yml` into `conformance.yml` and delete the
   separate full-conformance workflow.
