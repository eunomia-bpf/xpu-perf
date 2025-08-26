/**
 * Device Query
 * This sample queries the properties of the CUDA devices present in the system
 * From CUDA SDK Samples - Modified for CUDA 12.8 and 13.0 compatibility
 */

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

// Forward declaration
inline int _ConvertSMVer2Cores(int major, int minor);

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);
    printf(" CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
               (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if (deviceCount == 0) {
        printf("There are no available device(s) that support CUDA\n");
    } else {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev) {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
               driverVersion / 1000, (driverVersion % 100) / 10,
               runtimeVersion / 1000, (runtimeVersion % 100) / 10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n",
               deviceProp.major, deviceProp.minor);

        char msg[256];
        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem / 1048576.0f, (unsigned long long)deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors, (%3d) CUDA Cores/MP:     %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
               deviceProp.multiProcessorCount);

        printf("  Memory Bus Width:                              %d-bit\n",
               deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize) {
            printf("  L2 Cache Size:                                 %d bytes\n",
                   deviceProp.l2CacheSize);
        }

        printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n",
               deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          Yes with %d copy engine(s)\n",
               deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n",
               deviceProp.computePreemptionSupported ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n",
               deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n",
               deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n",
               deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n",
               deviceProp.ECCEnabled ? "Enabled" : "Disabled");
        printf("  Device supports Unified Addressing (UVA):      %s\n",
               deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Supports Cooperative Kernel Launch:            %s\n",
               deviceProp.cooperativeLaunch ? "Yes" : "No");
        printf("  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
               deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);
    }

    printf("\ndeviceQuery, CUDA Driver = CUDART, CUDA Driver Version = %d.%d, CUDA Runtime Version = %d.%d, NumDevs = %d\n",
           driverVersion / 1000, (driverVersion % 100) / 10, 
           runtimeVersion / 1000, (runtimeVersion % 100) / 10,
           deviceCount);
    
    printf("Result = PASS\n");

    return 0;
}

inline int _ConvertSMVer2Cores(int major, int minor)
{
    typedef struct {
        int SM;
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] = {
        {0x30, 192},
        {0x32, 192},
        {0x35, 192},
        {0x37, 192},
        {0x50, 128},
        {0x52, 128},
        {0x53, 128},
        {0x60, 64},
        {0x61, 128},
        {0x62, 128},
        {0x70, 64},
        {0x72, 64},
        {0x75, 64},
        {0x80, 64},
        {0x86, 128},
        {0x87, 128},
        {0x89, 128},
        {0x90, 128},
        {-1, -1}
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1) {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor)) {
            return nGpuArchCoresPerSM[index].Cores;
        }
        index++;
    }

    printf("MapSMtoCores for SM %d.%d is undefined. Default to use 64 Cores/SM\n", major, minor);
    return 64;
}