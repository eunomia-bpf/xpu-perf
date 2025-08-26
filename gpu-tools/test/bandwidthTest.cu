/**
 * Bandwidth Test
 * This test measures host to device and device to device copy bandwidth
 * for pageable and pinned memory of various sizes.
 * From CUDA SDK Samples
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

#define MEMCOPY_ITERATIONS 100
#define DEFAULT_SIZE       (32 * (1<<20))
#define DEFAULT_INCREMENT  (1<<22)

#define CACHE_CLEAR_SIZE (1<<24)

const char *sMemoryCopyKind[] = {
    "Host to Device",
    "Device to Host",
    "Device to Device"
};

const char *sMemoryMode[] = {
    "PAGEABLE",
    "PINNED"
};

void printResultsReadable(unsigned int *memSizes, double *bandwidths, 
                          unsigned int count, cudaMemcpyKind kind, 
                          int memMode, int iNumDevs)
{
    printf("Bandwidth Test %s Memory Transfers\n", sMemoryMode[memMode]);
    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    
    for (unsigned int i = 0; i < count; i++) {
        printf("   %u\t\t\t%s%.1f\n", memSizes[i], 
               (bandwidths[i] < 10000.0) ? "\t" : "",
               bandwidths[i]);
    }
    printf("\n");
}

void testBandwidth(unsigned int memSize, cudaMemcpyKind kind, int memMode)
{
    int iNumDevs = 0;
    checkCudaErrors(cudaGetDeviceCount(&iNumDevs));
    
    if (iNumDevs == 0) {
        printf("No CUDA-capable devices found.\n");
        exit(EXIT_FAILURE);
    }
    
    unsigned char *h_data = NULL;
    unsigned char *h_cacheClear = NULL;
    unsigned char *d_data = NULL;
    
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));
    
    checkCudaErrors(cudaMalloc((void **)&d_data, memSize));
    
    if (memMode == 1) {
        checkCudaErrors(cudaMallocHost((void **)&h_data, memSize));
        checkCudaErrors(cudaMallocHost((void **)&h_cacheClear, CACHE_CLEAR_SIZE));
    } else {
        h_data = (unsigned char *)malloc(memSize);
        h_cacheClear = (unsigned char *)malloc(CACHE_CLEAR_SIZE);
        
        if (!h_data || !h_cacheClear) {
            fprintf(stderr, "Failed to allocate host memory\n");
            exit(EXIT_FAILURE);
        }
    }
    
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        h_data[i] = (unsigned char)(i & 0xff);
    }
    
    for (unsigned int i = 0; i < CACHE_CLEAR_SIZE / sizeof(unsigned char); i++) {
        h_cacheClear[i] = (unsigned char)(i & 0xff);
    }
    
    checkCudaErrors(cudaEventRecord(start, 0));
    
    for (unsigned int i = 0; i < MEMCOPY_ITERATIONS; i++) {
        if (kind == cudaMemcpyHostToDevice) {
            checkCudaErrors(cudaMemcpy(d_data, h_data, memSize, cudaMemcpyHostToDevice));
        } else if (kind == cudaMemcpyDeviceToHost) {
            checkCudaErrors(cudaMemcpy(h_data, d_data, memSize, cudaMemcpyDeviceToHost));
        }
    }
    
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));
    
    float elapsedTimeInMs = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&elapsedTimeInMs, start, stop));
    
    double elapsedTimeInSec = elapsedTimeInMs / 1000.0;
    double bandwidth = ((double)memSize * (double)MEMCOPY_ITERATIONS) /
                      (elapsedTimeInSec * (double)(1<<20));
    
    unsigned int memSizes[] = {memSize};
    double bandwidths[] = {bandwidth};
    printResultsReadable(memSizes, bandwidths, 1, kind, memMode, iNumDevs);
    
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    
    if (memMode == 1) {
        checkCudaErrors(cudaFreeHost(h_data));
        checkCudaErrors(cudaFreeHost(h_cacheClear));
    } else {
        free(h_data);
        free(h_cacheClear);
    }
    
    checkCudaErrors(cudaFree(d_data));
}

int main(int argc, char **argv)
{
    printf("[Bandwidth Test] - Starting...\n");
    
    int device = 0;
    checkCudaErrors(cudaSetDevice(device));
    
    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, device));
    printf("Device %d: %s\n", device, deviceProp.name);
    
    unsigned int memSize = DEFAULT_SIZE;
    
    printf("\nQuick Mode\n");
    testBandwidth(memSize, cudaMemcpyHostToDevice, 0);
    testBandwidth(memSize, cudaMemcpyDeviceToHost, 0);
    
    printf("\nPinned Memory Transfers\n");
    testBandwidth(memSize, cudaMemcpyHostToDevice, 1);
    testBandwidth(memSize, cudaMemcpyDeviceToHost, 1);
    
    printf("\nTest passed\n");
    
    return 0;
}