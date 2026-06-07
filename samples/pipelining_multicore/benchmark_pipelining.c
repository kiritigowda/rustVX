/** @file benchmark_pipelining.c
 * Benchmark pipelining vs non-pipelining on a multi-scale graph.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <VX/vx.h>
#include <VX/vx_khr_pipelining.h>

#define WIDTH   1280
#define HEIGHT   720
#define ITERS     30
#define NUM_BUF    3

double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static vx_graph build_graph(vx_context ctx, vx_image in_ref, vx_image out_ref)
{
    vx_graph graph = vxCreateGraph(ctx);

    /* Three parallel branches from input */
    vx_image tmp_a = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image tmp_b = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image tmp_c = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);

    vx_node n_gauss = vxGaussian3x3Node(graph, in_ref, tmp_a);
    vx_node n_erode = vxErode3x3Node(graph, in_ref, tmp_b);
    vx_node n_dilate = vxDilate3x3Node(graph, in_ref, tmp_c);

    /* Fuse: AND all three */
    vx_image tmp_d = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_and1 = vxAndNode(graph, tmp_a, tmp_b, tmp_d);
    vx_node n_and2 = vxAndNode(graph, tmp_d, tmp_c, out_ref);

    /* Graph parameters */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_gauss, 0));
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_and2, 2));

    if (vxVerifyGraph(graph) != VX_SUCCESS) {
        fprintf(stderr, "Graph verification failed\n");
        vxReleaseGraph(&graph);
        return NULL;
    }
    return graph;
}

static double bench_non_pipelining(vx_context ctx)
{
    vx_image input  = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image output = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_graph graph = build_graph(ctx, input, output);
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
    vx_image in_bufs[NUM_BUF], out_bufs[NUM_BUF];
    for (int i = 0; i < NUM_BUF; i++) {
        in_bufs[i]  = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
        out_bufs[i] = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    }

    vx_graph graph = build_graph(ctx, in_bufs[0], out_bufs[0]);
    if (!graph) return -1.0;

    vx_graph_parameter_queue_params_t qp[2];
    qp[0].graph_parameter_index = 0;
    qp[0].refs_list_size = NUM_BUF;
    qp[0].refs_list = (vx_reference*)in_bufs;
    qp[1].graph_parameter_index = 1;
    qp[1].refs_list_size = NUM_BUF;
    qp[1].refs_list = (vx_reference*)out_bufs;

    vxSetGraphScheduleConfig(graph, VX_GRAPH_SCHEDULE_MODE_QUEUE_AUTO, 2, qp);
    vxScheduleGraph(graph);

    double t0 = now_ms();

    /* Prime pipeline */
    for (int i = 0; i < NUM_BUF; i++) {
        vxGraphParameterEnqueueReadyRef(graph, 0, (vx_reference*)&in_bufs[i], 1);
        vxGraphParameterEnqueueReadyRef(graph, 1, (vx_reference*)&out_bufs[i], 1);
    }

    /* Steady state */
    int idx = NUM_BUF;
    for (int i = 0; i < ITERS - NUM_BUF; i++) {
        vx_reference ref = NULL;
        vx_uint32 num = 0;
        vxGraphParameterDequeueDoneRef(graph, 1, &ref, 1, &num);
        vxGraphParameterEnqueueReadyRef(graph, 0, (vx_reference*)&in_bufs[idx % NUM_BUF], 1);
        vxGraphParameterEnqueueReadyRef(graph, 1, (vx_reference*)&out_bufs[idx % NUM_BUF], 1);
        idx++;
    }

    vxWaitGraph(graph);
    double t1 = now_ms();

    vxReleaseGraph(&graph);
    for (int i = 0; i < NUM_BUF; i++) {
        vxReleaseImage(&in_bufs[i]);
        vxReleaseImage(&out_bufs[i]);
    }
    return (t1 - t0) / ITERS;
}

int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    vx_context ctx = vxCreateContext();

    /* Pipelining extension check skipped — rustVX always includes it */

    printf("=== Pipelining vs Non-Pipelining Benchmark ===\n");
    printf("Graph: Input → [Gaussian | Erode | Dilate] → AND → AND → Output\n");
    printf("Resolution: %dx%d | Iterations: %d | Queue depth: %d\n\n",
           WIDTH, HEIGHT, ITERS, NUM_BUF);

    /* Warmup */
    printf("Warming up...\n");
    bench_non_pipelining(ctx);
    bench_pipelining(ctx);

    printf("[1/2] NON-PIPELINING (vxProcessGraph)...\n");
    double ms_non = bench_non_pipelining(ctx);
    double fps_non = 1000.0 / ms_non;
    printf("      %.3f ms/frame = %.2f FPS\n\n", ms_non, fps_non);

    printf("[2/2] PIPELINING (QUEUE_AUTO + enqueue/dequeue)...\n");
    double ms_pipe = bench_pipelining(ctx);
    double fps_pipe = 1000.0 / ms_pipe;
    printf("      %.3f ms/frame = %.2f FPS\n\n", ms_pipe, fps_pipe);

    double speedup = fps_pipe / fps_non;
    printf("=== Results ===\n");
    printf("Non-pipelining:  %.2f FPS\n", fps_non);
    printf("Pipelining:        %.2f FPS\n", fps_pipe);
    printf("Speedup:           %.2fx\n", speedup);

    if (speedup >= 1.5) printf("\n✓ Strong speedup: parallelism is effective\n");
    else if (speedup >= 1.2) printf("\n○ Moderate speedup\n");
    else printf("\n! Minimal speedup\n");

    vxReleaseContext(&ctx);
    return 0;
}
