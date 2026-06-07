/** 
 * @file pipelining_multicore.c
 * @brief OpenVX Pipelining Extension — Multicore Sample
 * 
 * This sample demonstrates how to use the OpenVX pipelining extension
 * with multicore (wave-based parallel) execution on a compute graph.
 * 
 * Requirements:
 *   - rustVX built with -DOPENVX_USE_PIPELINING=ON
 *   - OPENVX_PIPELINING_THREADS env var (optional)
 * 
 * Build:
 *   gcc -o pipelining_multicore pipelining_multicore.c -lopenvx -I/path/to/openvx/include
 * 
 * Run:
 *   ./pipelining_multicore                    # auto-detect thread count
 *   OPENVX_PIPELINING_THREADS=4 ./pipelining_multicore  # use 4 threads
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <VX/vx.h>

#define WIDTH   640
#define HEIGHT  480
#define ITERS   100

/* Simple user kernel: fill image with a constant value */
vx_status VX_CALLBACK fillKernel(vx_node node, const vx_reference *params, vx_uint32 num)
{
    (void)node;
    (void)num;
    vx_image out = (vx_image)params[0];
    vx_scalar val_s = (vx_scalar)params[1];
    vx_uint8 val = 0;
    vxCopyScalar(val_s, &val, VX_READ_ONLY, VX_MEMORY_TYPE_HOST);
    
    vx_rectangle_t rect = {0, 0, WIDTH, HEIGHT};
    vx_imagepatch_addressing_t addr;
    void *base = NULL;
    vx_map_id map_id;
    vxMapImagePatch(out, &rect, 0, &map_id, &addr, &base, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0);
    
    memset(base, val, addr.stride_y * HEIGHT);
    
    vxUnmapImagePatch(out, map_id);
    return VX_SUCCESS;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    vx_status status;
    vx_context context = vxCreateContext();
    
    /* Query pipelining extension availability */
    vx_bool pipelining = vx_false_e;
    vxQueryContext(context, VX_CONTEXT_EXTENSIONS, &pipelining, sizeof(pipelining));
    if (!pipelining) {
        fprintf(stderr, "OpenVX pipelining extension not available.\n");
        fprintf(stderr, "Build rustVX with: -DOPENVX_USE_PIPELINING=ON\n");
        return 1;
    }
    printf("✓ OpenVX Pipelining Extension available\n");
    
    /* Create a graph with parallel branches for multicore execution */
    vx_graph graph = vxCreateGraph(context);
    
    /* Graph parameters: input image + 4 output images (parallel branches) */
    vx_image input = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image out_a = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image out_b = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image out_c = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image out_d = vxCreateImage(context, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    /* Scalar parameter (fill value) */
    vx_uint8 fill_val = 128;
    vx_scalar scalar = vxCreateScalar(context, VX_TYPE_UINT8, &fill_val);
    
    /* Create user kernel for fill operation */
    vx_kernel kernel = vxAddUserKernel(context, "example.fill", VX_KERNEL_BASE(VX_ID_USER, 0) + 1,
                                        fillKernel, 2,
                                        NULL, NULL, NULL);
    if (kernel) {
        vxAddParameterToKernel(kernel, 0, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED);
        vxAddParameterToKernel(kernel, 1, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED);
        vxFinalizeKernel(kernel);
    }
    
    /* Build graph: input → [parallel branches] → outputs */
    /* Branch A: Gaussian blur */
    vx_image tmp_a = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_gauss = vxGaussian3x3Node(graph, input, tmp_a);
    
    /* Branch B: Box filter */
    vx_image tmp_b = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_box = vxBox3x3Node(graph, input, tmp_b);
    
    /* Branch C: Dilate */
    vx_image tmp_c = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_dilate = vxDilate3x3Node(graph, input, tmp_c);
    
    /* Branch D: Erode */
    vx_image tmp_d = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_erode = vxErode3x3Node(graph, input, tmp_d);
    
    /* Second wave: user kernel fills on each branch output */
    vx_node n_fill_a = vxCreateGenericNode(graph, kernel);
    vxSetParameterByIndex(n_fill_a, 0, (vx_reference)out_a);
    vxSetParameterByIndex(n_fill_a, 1, (vx_reference)scalar);
    
    vx_node n_fill_b = vxCreateGenericNode(graph, kernel);
    vxSetParameterByIndex(n_fill_b, 0, (vx_reference)out_b);
    vxSetParameterByIndex(n_fill_b, 1, (vx_reference)scalar);
    
    vx_node n_fill_c = vxCreateGenericNode(graph, kernel);
    vxSetParameterByIndex(n_fill_c, 0, (vx_reference)out_c);
    vxSetParameterByIndex(n_fill_c, 1, (vx_reference)scalar);
    
    vx_node n_fill_d = vxCreateGenericNode(graph, kernel);
    vxSetParameterByIndex(n_fill_d, 0, (vx_reference)out_d);
    vxSetParameterByIndex(n_fill_d, 1, (vx_reference)scalar);
    
    /* Configure graph parameters for pipelining */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_gauss, 0));   /* input */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_fill_a, 0));  /* out_a */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_fill_b, 0));  /* out_b */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_fill_c, 0));  /* out_c */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_fill_d, 0));  /* out_d */
    
    /* Verify graph — this computes topological waves for multicore execution */
    status = vxVerifyGraph(graph);
    if (status != VX_SUCCESS) {
        fprintf(stderr, "Graph verification failed: %d\n", status);
        return 1;
    }
    printf("✓ Graph verified (topological waves computed)\n");
    
    /* Enable pipelining with QUEUE_AUTO mode */
    vx_graph_parameter_queue_params_t queue_params[5];
    for (int i = 0; i < 5; i++) {
        queue_params[i].graph_parameter_index = i;
        queue_params[i].refs_list = NULL;  /* Will be set per enqueue */
        queue_params[i].refs_list_size = 1;
    }
    
    vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO,
                              5, queue_params);
    printf("✓ Pipelining mode set to QUEUE_AUTO\n");
    
    /* Schedule graph (starts background executor thread) */
    vxScheduleGraph(graph);
    printf("✓ Graph scheduled (executor thread started)\n");
    
    /* Warmup */
    printf("Warming up...\n");
    for (int i = 0; i < 10; i++) {
        vx_graph_parameter_enqueue_ready_ref(graph, 0, (vx_reference)input, 1);
        vx_reference out_refs[4] = {(vx_reference)out_a, (vx_reference)out_b,
                                    (vx_reference)out_c, (vx_reference)out_d};
        for (int j = 1; j < 5; j++) {
            vx_graph_parameter_enqueue_ready_ref(graph, j, out_refs[j-1], 1);
        }
    }
    
    /* Benchmark */
    printf("Running benchmark (%d iterations)...\n", ITERS);
    vx_uint64 t0 = vxGetTimestamp(context);
    
    for (int i = 0; i < ITERS; i++) {
        /* Enqueue input frame */
        vx_graph_parameter_enqueue_ready_ref(graph, 0, (vx_reference)input, 1);
        
        /* Enqueue output buffers */
        vx_reference out_refs[4] = {(vx_reference)out_a, (vx_reference)out_b,
                                    (vx_reference)out_c, (vx_reference)out_d};
        for (int j = 1; j < 5; j++) {
            vx_graph_parameter_enqueue_ready_ref(graph, j, out_refs[j-1], 1);
        }
    }
    
    /* Flush remaining frames */
    vxWaitGraph(graph);
    
    vx_uint64 t1 = vxGetTimestamp(context);
    double ms = (double)(t1 - t0) / 1000000.0;
    double fps = (ITERS * 1000.0) / ms;
    
    printf("\n=== Results ===\n");
    printf("Total time:  %.2f ms\n", ms);
    printf("Iterations:   %d\n", ITERS);
    printf("Throughput:   %.2f FPS\n", fps);
    printf("\nNotes:\n");
    printf("- Nodes in Wave 0 (Gaussian, Box, Dilate, Erode) execute in parallel\n");
    printf("- Nodes in Wave 1 (4× Fill) execute in parallel after Wave 0\n");
    printf("- Set OPENVX_PIPELINING_THREADS=N to control thread pool size\n");
    
    /* Cleanup */
    vxReleaseGraph(&graph);
    vxReleaseImage(&input);
    vxReleaseImage(&out_a);
    vxReleaseImage(&out_b);
    vxReleaseImage(&out_c);
    vxReleaseImage(&out_d);
    vxReleaseScalar(&scalar);
    vxRemoveKernel(kernel);
    vxReleaseContext(&context);
    
    return 0;
}
