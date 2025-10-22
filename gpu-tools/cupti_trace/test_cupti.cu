#include <stdio.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorSubtract(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] - b[i];
    }
}

int main() {
    int n = 1000;
    size_t size = n * sizeof(float);

    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Create CUDA graph
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;
    cudaGraphNode_t memcpyNode1, memcpyNode2, memcpyNode3;
    cudaGraphNode_t kernelNode1, kernelNode2;
    cudaStream_t stream;

    cudaStreamCreate(&stream);
    cudaGraphCreate(&graph, 0);

    // Add memcpy nodes for input data (H2D)
    cudaMemcpy3DParms memcpyParams1 = {0};
    memcpyParams1.srcPtr.ptr = h_a;
    memcpyParams1.dstPtr.ptr = d_a;
    memcpyParams1.extent.width = size;
    memcpyParams1.extent.height = 1;
    memcpyParams1.extent.depth = 1;
    memcpyParams1.kind = cudaMemcpyHostToDevice;
    cudaGraphAddMemcpyNode(&memcpyNode1, graph, NULL, 0, &memcpyParams1);

    cudaMemcpy3DParms memcpyParams2 = {0};
    memcpyParams2.srcPtr.ptr = h_b;
    memcpyParams2.dstPtr.ptr = d_b;
    memcpyParams2.extent.width = size;
    memcpyParams2.extent.height = 1;
    memcpyParams2.extent.depth = 1;
    memcpyParams2.kind = cudaMemcpyHostToDevice;
    cudaGraphAddMemcpyNode(&memcpyNode2, graph, NULL, 0, &memcpyParams2);

    // Add kernel nodes
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    void* kernelArgs1[] = {&d_a, &d_b, &d_c, &n};
    cudaKernelNodeParams kernelParams1 = {0};
    kernelParams1.func = (void*)vectorAdd;
    kernelParams1.gridDim = dim3(numBlocks, 1, 1);
    kernelParams1.blockDim = dim3(blockSize, 1, 1);
    kernelParams1.sharedMemBytes = 0;
    kernelParams1.kernelParams = kernelArgs1;
    kernelParams1.extra = NULL;

    cudaGraphNode_t deps1[] = {memcpyNode1, memcpyNode2};
    cudaGraphAddKernelNode(&kernelNode1, graph, deps1, 2, &kernelParams1);

    // Add second kernel node (vectorSubtract)
    void* kernelArgs2[] = {&d_a, &d_b, &d_c, &n};
    cudaKernelNodeParams kernelParams2 = {0};
    kernelParams2.func = (void*)vectorSubtract;
    kernelParams2.gridDim = dim3(numBlocks, 1, 1);
    kernelParams2.blockDim = dim3(blockSize, 1, 1);
    kernelParams2.sharedMemBytes = 0;
    kernelParams2.kernelParams = kernelArgs2;
    kernelParams2.extra = NULL;

    cudaGraphNode_t deps2[] = {kernelNode1};
    cudaGraphAddKernelNode(&kernelNode2, graph, deps2, 1, &kernelParams2);

    // Add memcpy node for output data (D2H)
    cudaMemcpy3DParms memcpyParams3 = {0};
    memcpyParams3.srcPtr.ptr = d_c;
    memcpyParams3.dstPtr.ptr = h_c;
    memcpyParams3.extent.width = size;
    memcpyParams3.extent.height = 1;
    memcpyParams3.extent.depth = 1;
    memcpyParams3.kind = cudaMemcpyDeviceToHost;

    cudaGraphNode_t deps3[] = {kernelNode2};
    cudaGraphAddMemcpyNode(&memcpyNode3, graph, deps3, 1, &memcpyParams3);

    // Instantiate and launch the graph
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    printf("Verification: h_c[0] = %f (expected -0), h_c[999] = %f (expected -999)\n",
           h_c[0], h_c[999]);

    // Cleanup
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
