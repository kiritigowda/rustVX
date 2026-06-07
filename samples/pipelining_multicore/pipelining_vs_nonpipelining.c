/**
 * @file pipelining_vs_nonpipelining.c
 * @brief Benchmark: Pipelining vs Non-Pipelining Throughput
 *
 * Compares frame throughput with and without the OpenVX pipelining extension.
 *
 * Pipeline:
 *   Input
 *    ├──→ Gaussian3x3 ──→ tmp_a
 *    ├──→ Erode3x3    ──→ tmp_b
 *    └──→ Dilate3x3   ──→ tmp_c
 *            └──→ AND(tmp_a, tmp_b) ──→ tmp_d
 *                    └──→ AND(tmp_d, tmp_c) ──→ Output
 *
 * Two modes:
 *   1. NON-PIPELINING:  vxProcessGraph (one frame at a time)
 *   2. PIPELINING:      vxScheduleGraph + enqueue/dequeue (overlap + multicore)
 *
 * Build:
 *   make OPENVX_INCLUDE=/path/to/include OPENVX_LIB=/path/to/lib
 *
 * Run:
 *   ./pipelining_vs_nonpipelining
 *   OPENVX_PIPELINING_THREADS=4 ./pipelining_vs_nonpipelining
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx.h>

#define WIDTH    640
#define HEIGHT   480
#define ITERS    60
#define NUM_BUF  3

double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

/* Build a graph with 3 parallel filter branches feeding into a chain of ANDs.
 * Returns the graph and populates gp_count with number of graph params. */
static vx_graph build_graph(vx_context ctx, int use_pipelining,
                               vx_image in_ref, vx_image out_ref,
                               vx_image **in_bufs, vx_image **out_bufs,
                               int *gp_count)
{
    vx_graph graph = vxCreateGraph(ctx);
    
    /* Virtual intermediates */
    vx_image tmp_a = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image tmp_b = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image tmp_c = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image tmp_d = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    
    /* Wave 0: three parallel 3x3 filters (all read input, no inter-dependency) */
    vx_node n_gauss = vxGaussian3x3Node(graph, in_ref, tmp_a);
    vx_node n_erode = vxErode3x3Node(graph,  in_ref, tmp_b);
    vx_node n_dilate = vxDilate3x3Node(graph, in_ref, tmp_c);
    
    /* Wave 1: combine results (tmp_a & tmp_b & tmp_c) */
    vx_node n_and1 = vxAndNode(graph, tmp_a, tmp_b, tmp_d);
    vx_node n_and2 = vxAndNode(graph, tmp_d, tmp_c, out_ref);
    
    /* Graph parameters: input + output */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_gauss, 0));
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_and2,  2));
    *gp_count = 2;
    
    if (vxVerifyGraph(graph) != VX_SUCCESS) {
        fprintf(stderr, "Graph verification failed\n");
        vxReleaseGraph(&graph);
        return NULL;
    }
    
    if (use_pipelining) {
        /* Allocate extra buffer pairs for pipelining */
        *in_bufs  = malloc(NUM_BUF * sizeof(vx_image));
        *out_bufs = malloc(NUM_BUF * sizeof(vx_image));
        for (int i = 0; i < NUM_BUF; i++) {
            (*in_bufs)[i]  = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
            (*out_bufs)[i] = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
        }
        
        vx_graph_parameter_queue_params_t qp[2];
        qp[0].graph_parameter_index = 0;
        qp[0].refs_list_size = NUM_BUF;
        qp[0].refs_list = (vx_reference*)(*in_bufs);
        qp[1].graph_parameter_index = 1;
        qp[1].refs_list_size = NUM_BUF;
        qp[1].refs_list = (vx_reference*)(*out_bufs);
        
        vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO, 2, qp);
    }
    
    return graph;
}

static double bench_non_pipelining(vx_context ctx)
{
    vx_image input  = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    int gp_count = 0;
    vx_graph graph = build_graph(ctx, 0, input, output, NULL, NULL, &gp_count);
    if (!graph) return -1.0;
    
    double t0 = now_ms();
    for (int i = 0; i < ITERS; i++) {
        vxProcessGraph(graph);
    }
    double t1 = now_ms();
    
    vxReleaseGraph(&graph);
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    return (t1 - t0) / ITERS;
}

static double bench_pipelining(vx_context ctx)
{
    vx_image input0 = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image output0 = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    int gp_count = 0;
    vx_image *in_bufs = NULL, *out_bufs = NULL;
    vx_graph graph = build_graph(ctx, 1, input0, output0, &in_bufs, &out_bufs, &gp_count);
    if (!graph) return -1.0;
    
    /* Start executor thread */
    vxScheduleGraph(graph);
    
    double t0 = now_ms();
    
    /* Prime pipeline */
    for (int i = 0; i < NUM_BUF; i++) {
        vx_graph_parameter_enqueue_ready_ref(graph, 0, (vx_reference)&in_bufs[i], 1);
        vx_graph_parameter_enqueue_ready_ref(graph, 1, (vx_reference)&out_bufs[i], 1);
    }
    
    /* Steady state: recycle completed buffers */
    int idx = NUM_BUF;
    for (int i = 0; i < ITERS - NUM_BUF; i++) {
        vx_reference ref = NULL;
        vx_uint32 num = 0;
        vx_graph_parameter_dequeue_done_ref(graph, 1, &ref, 1, &num);
        vx_graph_parameter_enqueue_ready_ref(graph, 0, (vx_reference)&in_bufs[idx % NUM_BUF], 1);
        vx_graph_parameter_enqueue_ready_ref(graph, 1, (vx_reference)&out_bufs[idx % NUM_BUF], 1);
        idx++;
    }
    
    /* Drain */
    vxWaitGraph(graph);
    
    double t1 = now_ms();
    
    vxReleaseGraph(&graph);
    vxReleaseImage(&input0);
    vxReleaseImage(&output0);
    for (int i = 0; i < NUM_BUF; i++) {
        vxReleaseImage(&in_bufs[i]);
        vxReleaseImage(&out_bufs[i]);
    }
    free(in_bufs);
    free(out_bufs);
    return (t1 - t0) / ITERS;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    vx_context ctx = vxCreateContext();
    
    vx_bool pipelining = vx_false_e;
    vxQueryContext(ctx, VX_CONTEXT_EXTENSIONS, &pipelining, sizeof(pipelining));
    if (!pipelining) {
        fprintf(stderr, "Pipelining extension not available. Build with -DOPENVX_USE_PIPELINING=ON\n");
        return 1;
    }
    
    printf("=== OpenVX Pipelining Performance Comparison ===\n\n");
    printf("Graph:  Input → [Gaussian | Erode | Dilate] → AND → AND → Output\n");
    printf("        (3 parallel branches, 2 sequential combines)\n");
    printf("Resolution: %dx%d | Iterations: %d | Queue depth: %d\n\n",
           WIDTH, HEIGHT, ITERS, NUM_BUF);
    
    /* Mode 1: Non-pipelining */
    printf("[1/2] NON-PIPELINING (vxProcessGraph)...\n");
    double ms_non = bench_non_pipelining(ctx);
    if (ms_non < 0) return 1;
    double fps_non = 1000.0 / ms_non;
    printf("      %.3f ms/frame = %.2f FPS\n\n", ms_non, fps_non);
    
    /* Mode 2: Pipelining */
    printf("[2/2] PIPELINING (QUEUE_AUTO + enqueue/dequeue)...\n");
    double ms_pipe = bench_pipelining(ctx);
    if (ms_pipe < 0) return 1;
    double fps_pipe = 1000.0 / ms_pipe;
    printf("      %.3f ms/frame = %.2f FPS\n\n", ms_pipe, fps_pipe);
    
    /* Summary */
    double speedup = fps_pipe / fps_non;
    printf("=== Summary ===\n");
    printf("Non-pipelining throughput:  %.2f FPS\n", fps_non);
    printf("Pipelining throughput:        %.2f FPS\n", fps_pipe);
    printf("Throughput speedup:           %.2fx\n", speedup);
    printf("Latency reduction:            %.1f%%\n", (1.0 - ms_pipe/ms_non) * 100.0);
    
    if (speedup >= 1.5) {
        printf("\n✓ Strong speedup: pipelining + multicore are both active\n");
    } else if (speedup >= 1.2) {
        printf("\n○ Moderate speedup: mainly from execution overlap\n");
    } else {
        printf("\n! Minimal speedup: graph may be too small for parallel benefit\n");
    }
    
    vxReleaseContext(&ctx);
    return 0;
}
