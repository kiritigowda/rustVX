/**
 * @file multiscale_feature_extraction.c
 * @brief Real-World Sample: Multi-Scale Feature Extraction Pipeline
 *
 * Simulates a computer vision preprocessing stage used in object detection
 * or tracking systems.  Extracts edges at three scales in parallel, then
 * fuses them into a single confidence map.
 *
 * Pipeline:
 *   Input Image (1920×1080 RGB)
 *      ├──→ ColorConvert → Y plane (grayscale)
 *      │       ├──→ Gaussian3x3 → tmp_0
 *      │       ├──→ HalfScaleGaussian → half
 *      │       │       └──→ Gaussian3x3 → tmp_1
 *      │       └──→ HalfScaleGaussian(half) → quarter
 *      │               └──→ Gaussian3x3 → tmp_2
 *      │
 *      └──→ [Parallel Wave]
 *              ├──→ Canny(tmp_0)  → edges_0
 *              ├──→ Canny(tmp_1)  → edges_1
 *              └──→ Canny(tmp_2)  → edges_2
 *                      └──→ [Fuse]
 *                              ├──→ Or(edges_0, edges_1) → fuse_a
 *                              └──→ Or(fuse_a, edges_2) → Output
 *
 * Why this is realistic:
 *   - Multi-scale processing is standard in detection pipelines (YOLO, SSD)
 *   - Three Canny detectors run in parallel on independent scales
 *   - The wave-based executor maps naturally to CPU cores
 *
 * Build:
 *   make OPENVX_INCLUDE=/path/to/include OPENVX_LIB=/path/to/lib
 *
 * Run:
 *   ./multiscale_feature_extraction
 *   OPENVX_PIPELINING_THREADS=4 ./multiscale_feature_extraction
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <vx/vx.h>

#define WIDTH       1920
#define HEIGHT      1080
#define ITERS       30
#define NUM_BUF     3

double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static vx_graph build_multiscale_graph(vx_context ctx,
                                        vx_image input_rgb,
                                        vx_image output,
                                        int use_pipelining,
                                        vx_image **in_bufs,
                                        vx_image **out_bufs)
{
    vx_graph graph = vxCreateGraph(ctx);

    /* Convert RGB → Y (grayscale).  In a real app you'd also keep U/V. */
    vx_image gray = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_color = vxColorConvertNode(graph, input_rgb, gray);
    (void)n_color; /* graph param registered below */

    /* Scale pyramid */
    vx_image half   = vxCreateVirtualImage(graph, WIDTH/2, HEIGHT/2, VX_DF_IMAGE_U8);
    vx_image quarter = vxCreateVirtualImage(graph, WIDTH/4, HEIGHT/4, VX_DF_IMAGE_U8);
    vx_node n_half   = vxHalfScaleGaussianNode(graph, gray, half, 3);
    vx_node n_quarter = vxHalfScaleGaussianNode(graph, half, quarter, 3);
    (void)n_half; (void)n_quarter;

    /* Blur each scale (reduces noise before edge detection) */
    vx_image tmp_0 = vxCreateVirtualImage(graph, WIDTH,     HEIGHT,     VX_DF_IMAGE_U8);
    vx_image tmp_1 = vxCreateVirtualImage(graph, WIDTH/2,   HEIGHT/2,   VX_DF_IMAGE_U8);
    vx_image tmp_2 = vxCreateVirtualImage(graph, WIDTH/4,   HEIGHT/4,   VX_DF_IMAGE_U8);

    vx_node n_blur0 = vxGaussian3x3Node(graph, gray,    tmp_0);
    vx_node n_blur1 = vxGaussian3x3Node(graph, half,    tmp_1);
    vx_node n_blur2 = vxGaussian3x3Node(graph, quarter, tmp_2);

    /* Parallel wave: Canny edge detection at each scale */
    vx_image edges_0 = vxCreateVirtualImage(graph, WIDTH,     HEIGHT,     VX_DF_IMAGE_U8);
    vx_image edges_1 = vxCreateVirtualImage(graph, WIDTH/2,   HEIGHT/2,   VX_DF_IMAGE_U8);
    vx_image edges_2 = vxCreateVirtualImage(graph, WIDTH/4,   HEIGHT/4,   VX_DF_IMAGE_U8);

    vx_threshold thresh0 = vxCreateThreshold(ctx, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vx_threshold thresh1 = vxCreateThreshold(ctx, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vx_threshold thresh2 = vxCreateThreshold(ctx, VX_THRESHOLD_TYPE_RANGE, VX_TYPE_UINT8);
    vxSetThresholdAttribute(thresh0, VX_THRESHOLD_THRESHOLD_LOWER, &(vx_int32){80},  sizeof(vx_int32));
    vxSetThresholdAttribute(thresh0, VX_THRESHOLD_THRESHOLD_UPPER, &(vx_int32){150}, sizeof(vx_int32));
    vxSetThresholdAttribute(thresh1, VX_THRESHOLD_THRESHOLD_LOWER, &(vx_int32){60},  sizeof(vx_int32));
    vxSetThresholdAttribute(thresh1, VX_THRESHOLD_THRESHOLD_UPPER, &(vx_int32){120}, sizeof(vx_int32));
    vxSetThresholdAttribute(thresh2, VX_THRESHOLD_THRESHOLD_LOWER, &(vx_int32){40},  sizeof(vx_int32));
    vxSetThresholdAttribute(thresh2, VX_THRESHOLD_THRESHOLD_UPPER, &(vx_int32){100}, sizeof(vx_int32));

    vx_node n_canny0 = vxCannyEdgeDetectorNode(graph, tmp_0, thresh0, 3, VX_NORM_L1, edges_0);
    vx_node n_canny1 = vxCannyEdgeDetectorNode(graph, tmp_1, thresh1, 3, VX_NORM_L1, edges_1);
    vx_node n_canny2 = vxCannyEdgeDetectorNode(graph, tmp_2, thresh2, 3, VX_NORM_L1, edges_2);
    (void)n_canny0; (void)n_canny1; (void)n_canny2;

    /* Fuse: OR all three edge maps (need to resize smaller maps to full resolution) */
    /* For simplicity, just OR the full-res edges with resized versions */
    vx_image edges_1_up = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image edges_2_up = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_scale1 = vxScaleImageNode(graph, edges_1, edges_1_up, VX_INTERPOLATION_BILINEAR);
    vx_node n_scale2 = vxScaleImageNode(graph, edges_2, edges_2_up, VX_INTERPOLATION_BILINEAR);
    (void)n_scale1; (void)n_scale2;

    vx_image fuse_a = vxCreateVirtualImage(graph, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_node n_or1 = vxOrNode(graph, edges_0, edges_1_up, fuse_a);
    vx_node n_or2 = vxOrNode(graph, fuse_a, edges_2_up, output);
    (void)n_or1; (void)n_or2;

    /* Graph parameters */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_color, 0)); /* input_rgb */
    vxAddParameterToGraph(graph, (vx_parameter)vxGetParameterByIndex(n_or2,   2)); /* output    */

    if (vxVerifyGraph(graph) != VX_SUCCESS) {
        fprintf(stderr, "Graph verification failed\n");
        vxReleaseGraph(&graph);
        return NULL;
    }

    if (use_pipelining) {
        *in_bufs  = malloc(NUM_BUF * sizeof(vx_image));
        *out_bufs = malloc(NUM_BUF * sizeof(vx_image));
        for (int i = 0; i < NUM_BUF; i++) {
            (*in_bufs)[i]  = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_RGB);
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

    vxReleaseThreshold(&thresh0);
    vxReleaseThreshold(&thresh1);
    vxReleaseThreshold(&thresh2);

    return graph;
}

static double bench(vx_context ctx, int use_pipelining)
{
    vx_image input0  = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_RGB);
    vx_image output0 = vxCreateImage(ctx, WIDTH, HEIGHT, VX_DF_IMAGE_U8);
    vx_image *in_bufs = NULL, *out_bufs = NULL;

    vx_graph graph = build_multiscale_graph(ctx, input0, output0, use_pipelining, &in_bufs, &out_bufs);
    if (!graph) return -1.0;

    double ms_per_frame = 0.0;

    if (!use_pipelining) {
        /* Non-pipelining: process one frame at a time */
        double t0 = now_ms();
        for (int i = 0; i < ITERS; i++) {
            vxProcessGraph(graph);
        }
        double t1 = now_ms();
        ms_per_frame = (t1 - t0) / ITERS;
    } else {
        /* Pipelining: overlapping execution with enqueue/dequeue */
        vxScheduleGraph(graph);

        double t0 = now_ms();

        /* Prime pipeline */
        for (int i = 0; i < NUM_BUF; i++) {
            vx_graph_parameter_enqueue_ready_ref(graph, 0, (vx_reference)&in_bufs[i], 1);
            vx_graph_parameter_enqueue_ready_ref(graph, 1, (vx_reference)&out_bufs[i], 1);
        }

        /* Steady state */
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
        ms_per_frame = (t1 - t0) / ITERS;
    }

    vxReleaseGraph(&graph);
    vxReleaseImage(&input0);
    vxReleaseImage(&output0);
    if (in_bufs) {
        for (int i = 0; i < NUM_BUF; i++) {
            vxReleaseImage(&in_bufs[i]);
            vxReleaseImage(&out_bufs[i]);
        }
        free(in_bufs);
        free(out_bufs);
    }
    return ms_per_frame;
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

    printf("=== Multi-Scale Feature Extraction (Real-World Pipeline) ===\n\n");
    printf("Graph:\n");
    printf("  RGB Input → ColorConvert → Y plane\n");
    printf("              ├──→ Gaussian3x3 ──→ Canny(full)   ──┐\n");
    printf("              ├──→ HalfScale → Gaussian3x3 ──→ Canny(half) ──→ ScaleUp ──┤→ OR → OR → Output\n");
    printf("              └──→ HalfScale → HalfScale → Gaussian3x3 ──→ Canny(quarter) ──→ ScaleUp ──┘\n\n");
    printf("Resolution: %dx%d RGB | Iterations: %d | Queue depth: %d\n\n",
           WIDTH, HEIGHT, ITERS, NUM_BUF);

    /* Warmup to stabilise caches */
    printf("Warming up...\n");
    bench(ctx, 0);
    bench(ctx, 1);

    /* Benchmark non-pipelining */
    printf("[1/2] NON-PIPELINING (vxProcessGraph)...\n");
    double ms_non = bench(ctx, 0);
    if (ms_non < 0) return 1;
    double fps_non = 1000.0 / ms_non;
    printf("      %.3f ms/frame = %.2f FPS\n\n", ms_non, fps_non);

    /* Benchmark pipelining */
    printf("[2/2] PIPELINING (QUEUE_AUTO + enqueue/dequeue)...\n");
    double ms_pipe = bench(ctx, 1);
    if (ms_pipe < 0) return 1;
    double fps_pipe = 1000.0 / ms_pipe;
    printf("      %.3f ms/frame = %.2f FPS\n\n", ms_pipe, fps_pipe);

    /* Summary */
    double speedup = fps_pipe / fps_non;
    printf("=== Summary ===\n");
    printf("Non-pipelining throughput:  %.2f FPS\n", fps_non);
    printf("Pipelining throughput:        %.2f FPS\n", fps_pipe);
    printf("Throughput speedup:         %.2fx\n", speedup);

    if (speedup >= 1.5) {
        printf("\n✓ Strong speedup: wave parallelism is working across CPU cores\n");
    } else if (speedup >= 1.2) {
        printf("\n○ Moderate speedup: mainly from overlapping frame execution\n");
    } else {
        printf("\n! Minimal speedup: the graph may be bandwidth-limited\n");
    }

    printf("\nTry: OPENVX_PIPELINING_THREADS=4 %s\n", argv[0]);

    vxReleaseContext(&ctx);
    return 0;
}
