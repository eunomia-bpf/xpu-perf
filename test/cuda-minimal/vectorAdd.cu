/**
 * Vector addition: C = A + B
 * This sample is a simple vector addition application to demonstrate CUDA kernels
 * From CUDA SDK Samples
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <sys/time.h>

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

double get_time_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main(void)
{
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    double target_runtime = 10.0; // Target 10 seconds
    int iteration = 0;

    printf("[Vector addition running for ~%.0f seconds]\n", target_runtime);
    printf("Elements per iteration: %d\n", numElements);

    double start_time = get_time_seconds();
    double elapsed = 0.0;

    while (elapsed < target_runtime) {
        float *h_A = (float *)malloc(size);
        float *h_B = (float *)malloc(size);
        float *h_C = (float *)malloc(size);

        if (h_A == NULL || h_B == NULL || h_C == NULL)
        {
            fprintf(stderr, "Failed to allocate host vectors!\n");
            exit(EXIT_FAILURE);
        }

        // Initialize input vectors
        for (int i = 0; i < numElements; ++i)
        {
            h_A[i] = rand() / (float)RAND_MAX;
            h_B[i] = rand() / (float)RAND_MAX;
        }

        float *d_A = NULL;
        float *d_B = NULL;
        float *d_C = NULL;

        cudaMalloc((void **)&d_A, size);
        cudaMalloc((void **)&d_B, size);
        cudaMalloc((void **)&d_C, size);

        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

        // Launch kernel
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

        // Verification only on first iteration to save time
        if (iteration == 0) {
            for (int i = 0; i < 100; ++i)
            {
                if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
                {
                    fprintf(stderr, "Result verification failed at element %d!\n", i);
                    fprintf(stderr, "Expected: %f, Got: %f\n", h_A[i] + h_B[i], h_C[i]);
                    exit(EXIT_FAILURE);
                }
            }
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        free(h_A);
        free(h_B);
        free(h_C);

        iteration++;
        elapsed = get_time_seconds() - start_time;

        // Print progress every 100 iterations or every 5 seconds
        static int last_print_sec = -1;
        int current_sec = (int)elapsed;
        if (iteration % 100 == 0 || (current_sec != last_print_sec && current_sec % 5 == 0)) {
            printf("Iteration %d, Elapsed: %.1fs / %.0fs\n", iteration, elapsed, target_runtime);
            last_print_sec = current_sec;
        }
    }

    printf("\nCompleted %d iterations in %.2f seconds\n", iteration, elapsed);
    printf("Test PASSED\n");
    printf("Done\n");
    return 0;
}