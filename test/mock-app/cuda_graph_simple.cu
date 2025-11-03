#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Simple vector scaling kernel
__global__ void vectorScale(float *A, float scalar, int N) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N) {
        A[i] *= scalar;
    }
}

int main(int argc, char **argv) {
    printf("=== CUDA Graph Simple Example ===\n\n");

    // Parse arguments
    int N = 1 << 20;  // 1M elements
    int iterations = 10;

    if (argc > 1) {
        iterations = atoi(argv[1]);
    }
    if (argc > 2) {
        N = atoi(argv[2]);
    }

    printf("Configuration:\n");
    printf("  Vector size: %d elements (%.2f MB)\n", N, N * sizeof(float) / 1e6);
    printf("  Iterations: %d\n", iterations);
    printf("\n");

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launch configuration:\n");
    printf("  Threads per block: %d\n", threadsPerBlock);
    printf("  Blocks per grid: %d\n", blocksPerGrid);
    printf("\n");

    // =========================================================================
    // PART 1: Without CUDA Graph (baseline)
    // =========================================================================
    printf("--- Running WITHOUT CUDA Graph ---\n");

    cudaEvent_t start_no_graph, stop_no_graph;
    CHECK_CUDA(cudaEventCreate(&start_no_graph));
    CHECK_CUDA(cudaEventCreate(&stop_no_graph));

    CHECK_CUDA(cudaEventRecord(start_no_graph));

    for (int i = 0; i < iterations; i++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        vectorScale<<<blocksPerGrid, threadsPerBlock>>>(d_C, 0.5f, N);
    }

    CHECK_CUDA(cudaEventRecord(stop_no_graph));
    CHECK_CUDA(cudaEventSynchronize(stop_no_graph));

    float time_no_graph = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_no_graph, start_no_graph, stop_no_graph));

    printf("Time without graph: %.3f ms\n", time_no_graph);
    printf("Average per iteration: %.3f ms\n", time_no_graph / iterations);
    printf("\n");

    // =========================================================================
    // PART 2: With CUDA Graph
    // =========================================================================
    printf("--- Running WITH CUDA Graph ---\n");

    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaStream_t stream;

    CHECK_CUDA(cudaStreamCreate(&stream));

    // Begin graph capture
    printf("Capturing CUDA Graph...\n");
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));

    // Launch kernels in the stream (these will be captured)
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_A, d_B, d_C, N);
    vectorScale<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_C, 0.5f, N);

    // End capture
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // Instantiate the graph
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));
    printf("Graph captured and instantiated!\n");

    // Get graph info
    size_t numNodes = 0;
    CHECK_CUDA(cudaGraphGetNodes(graph, NULL, &numNodes));
    printf("Graph contains %zu nodes\n", numNodes);
    printf("\n");

    // Warm up
    CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    cudaEvent_t start_graph, stop_graph;
    CHECK_CUDA(cudaEventCreate(&start_graph));
    CHECK_CUDA(cudaEventCreate(&stop_graph));

    CHECK_CUDA(cudaEventRecord(start_graph, stream));

    // Launch the graph multiple times
    for (int i = 0; i < iterations; i++) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }

    CHECK_CUDA(cudaEventRecord(stop_graph, stream));
    CHECK_CUDA(cudaEventSynchronize(stop_graph));

    float time_graph = 0;
    CHECK_CUDA(cudaEventElapsedTime(&time_graph, start_graph, stop_graph));

    printf("Time with graph: %.3f ms\n", time_graph);
    printf("Average per iteration: %.3f ms\n", time_graph / iterations);
    printf("\n");

    // =========================================================================
    // Results comparison
    // =========================================================================
    printf("=== Performance Comparison ===\n");
    printf("Without graph: %.3f ms\n", time_no_graph);
    printf("With graph:    %.3f ms\n", time_graph);
    printf("Speedup:       %.2fx\n", time_no_graph / time_graph);
    printf("Overhead saved: %.3f ms (%.1f%%)\n",
           time_no_graph - time_graph,
           100.0 * (time_no_graph - time_graph) / time_no_graph);
    printf("\n");

    // Verify results
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool correct = true;
    float expected = (h_A[0] + h_B[0]) * 0.5f;
    for (int i = 0; i < N; i++) {
        if (fabs(h_C[i] - expected) > 1e-5) {
            printf("Verification failed at index %d: expected %.2f, got %.2f\n",
                   i, expected, h_C[i]);
            correct = false;
            break;
        }
    }

    if (correct) {
        printf("✓ Results verified successfully!\n");
    }

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start_no_graph));
    CHECK_CUDA(cudaEventDestroy(stop_no_graph));
    CHECK_CUDA(cudaEventDestroy(start_graph));
    CHECK_CUDA(cudaEventDestroy(stop_graph));

    free(h_A);
    free(h_B);
    free(h_C);

    printf("\nDone!\n");

    return 0;
}
