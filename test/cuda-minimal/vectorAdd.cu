/**
 * Vector addition: C = A + B
 * This sample is a simple vector addition application to demonstrate CUDA kernels
 * From CUDA SDK Samples
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

int main(void)
{
    // Increase size and add iterations to run for ~30 seconds
    int numElements = 10000000;  // 10M elements (40MB per array)
    int iterations = 15000;       // Run many iterations for ~30s
    size_t size = numElements * sizeof(float);

    printf("[Vector addition of %d elements, %d iterations]\n", numElements, iterations);
    printf("Target runtime: ~30 seconds\n");

    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize input vectors
    printf("Initializing input vectors...\n");
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    printf("Allocating device memory...\n");
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    printf("Copy input data from the host memory to the CUDA device\n");
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel config: %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    printf("Starting %d iterations...\n", iterations);

    // Run many iterations to increase runtime
    for (int iter = 0; iter < iterations; iter++) {
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

        // Print progress every 1500 iterations
        if ((iter + 1) % 1500 == 0) {
            cudaDeviceSynchronize();
            printf("Completed iteration %d/%d\n", iter + 1, iterations);
        }
    }

    // Synchronize to ensure all kernels complete
    cudaDeviceSynchronize();
    printf("All iterations completed\n");

    printf("Copy output data from the CUDA device to the host memory\n");
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results (check first 1000 elements to save time)
    printf("Verifying results...\n");
    int checkElements = (numElements < 1000) ? numElements : 1000;
    for (int i = 0; i < checkElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}