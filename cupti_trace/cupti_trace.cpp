/*
 * Simplified CUPTI Trace Injection - Single File Version
 *
 * This is a simplified version of cupti_trace_injection.cpp that:
 * - Includes all necessary code in a single file (no external headers)
 * - Tracks runtime API calls and kernel executions
 * - Prints activity records to stdout or a file
 * - Supports basic configuration via environment variables
 *
 * Usage:
 *   CUDA_INJECTION64_PATH=./libcupti_trace_simple.so \
 *   CUPTI_TRACE_OUTPUT_FILE=/tmp/trace.txt \
 *   ./your_cuda_app
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <mutex>
#include <map>

// CUDA and CUPTI headers
#include <cuda.h>
#include <cupti.h>

#ifndef _WIN32
#include <pthread.h>
#include <unistd.h>
#endif

// ============================================================================
// Error Handling Macros
// ============================================================================

#define CUPTI_CALL(call)                                                       \
do {                                                                           \
    CUptiResult _status = call;                                                \
    if (_status != CUPTI_SUCCESS) {                                            \
        const char *errstr;                                                    \
        cuptiGetResultString(_status, &errstr);                                \
        fprintf(stderr, "CUPTI Error: %s:%d: %s\n", __FILE__, __LINE__, errstr); \
        exit(EXIT_FAILURE);                                                    \
    }                                                                          \
} while (0)

#define BUF_SIZE (32 * 1024 * 1024)  // 32 MB buffer size
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align) \
    (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

// ============================================================================
// Global State
// ============================================================================

// Data structures for graph node tracking
typedef struct ApiData_st {
    const char *pFunctionName;
    uint32_t correlationId;
} ApiData;

typedef std::map<uint64_t, ApiData> NodeIdApiDataMap;
static NodeIdApiDataMap nodeIdCorrelationMap;

struct GlobalState {
    CUpti_SubscriberHandle subscriberHandle;
    FILE *outputFile;
    int tracingEnabled;
    std::mutex initMutex;
    int initialized;
};

static GlobalState g_state = {0};

// ============================================================================
// Helper Functions
// ============================================================================

static const char* GetActivityKindString(CUpti_ActivityKind kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL: return "KERNEL";
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: return "CONCURRENT_KERNEL";
        case CUPTI_ACTIVITY_KIND_MEMCPY: return "MEMCPY";
        case CUPTI_ACTIVITY_KIND_MEMSET: return "MEMSET";
        case CUPTI_ACTIVITY_KIND_RUNTIME: return "RUNTIME";
        case CUPTI_ACTIVITY_KIND_DRIVER: return "DRIVER";
        case CUPTI_ACTIVITY_KIND_OVERHEAD: return "OVERHEAD";
        case CUPTI_ACTIVITY_KIND_MEMCPY2: return "MEMCPY2";
        case CUPTI_ACTIVITY_KIND_MEMORY2: return "MEMORY2";
        default: return "UNKNOWN";
    }
}

static const char* GetMemcpyKindString(uint8_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD: return "HtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH: return "DtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD: return "DtoD";
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH: return "HtoH";
        case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP: return "PtoP";
        default: return "UNKNOWN";
    }
}

static const char* GetMemoryKindString(uint8_t kind) {
    switch (kind) {
        case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN: return "UNKNOWN";
        case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE: return "PAGEABLE";
        case CUPTI_ACTIVITY_MEMORY_KIND_PINNED: return "PINNED";
        case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE: return "DEVICE";
        case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY: return "ARRAY";
        case CUPTI_ACTIVITY_MEMORY_KIND_MANAGED: return "MANAGED";
        default: return "UNKNOWN";
    }
}

static const char* GetMemoryOperationTypeString(CUpti_ActivityMemoryOperationType type) {
    switch (type) {
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_INVALID: return "INVALID";
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_ALLOCATION: return "ALLOCATE";
        case CUPTI_ACTIVITY_MEMORY_OPERATION_TYPE_RELEASE: return "RELEASE";
        default: return "UNKNOWN";
    }
}

static const char* GetMemoryPoolTypeString(CUpti_ActivityMemoryPoolType type) {
    switch (type) {
        case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_INVALID: return "INVALID";
        case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL: return "LOCAL";
        case CUPTI_ACTIVITY_MEMORY_POOL_TYPE_IMPORTED: return "IMPORTED";
        default: return "UNKNOWN";
    }
}

// ============================================================================
// Activity Record Printing
// ============================================================================

static void PrintActivity(CUpti_Activity *record) {
    FILE *out = g_state.outputFile;
    if (!out) return;

    switch (record->kind) {
        case CUPTI_ACTIVITY_KIND_KERNEL:
        case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
            CUpti_ActivityKernel9 *kernel = (CUpti_ActivityKernel9 *)record;
            fprintf(out, "%s [%llu, %llu] duration=%llu ns, name=\"%s\", correlationId=%u\n",
                    GetActivityKindString(kernel->kind),
                    (unsigned long long)kernel->start,
                    (unsigned long long)kernel->end,
                    (unsigned long long)(kernel->end - kernel->start),
                    kernel->name ? kernel->name : "<unknown>",
                    kernel->correlationId);
            fprintf(out, "  grid=[%u,%u,%u], block=[%u,%u,%u], "
                    "sharedMem(static=%u,dynamic=%u), deviceId=%u, streamId=%u",
                    kernel->gridX, kernel->gridY, kernel->gridZ,
                    kernel->blockX, kernel->blockY, kernel->blockZ,
                    kernel->staticSharedMemory, kernel->dynamicSharedMemory,
                    kernel->deviceId, kernel->streamId);
            if (kernel->graphNodeId != 0) {
                fprintf(out, ", graphId=%u, graphNodeId=%llu",
                        kernel->graphId, (unsigned long long)kernel->graphNodeId);
                // Check if we have API info for this graph node
                NodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(kernel->graphNodeId);
                if (it != nodeIdCorrelationMap.end()) {
                    fprintf(out, "\n  Graph node created by API: %s (correlationId=%u)",
                            it->second.pFunctionName, it->second.correlationId);
                }
            }
            fprintf(out, "\n");
            break;
        }

        case CUPTI_ACTIVITY_KIND_MEMCPY: {
            CUpti_ActivityMemcpy6 *memcpy = (CUpti_ActivityMemcpy6 *)record;
            fprintf(out, "%s \"%s\" [%llu, %llu] duration=%llu ns, size=%llu bytes, "
                    "srcKind=%s, dstKind=%s, correlationId=%u\n",
                    GetActivityKindString(memcpy->kind),
                    GetMemcpyKindString(memcpy->copyKind),
                    (unsigned long long)memcpy->start,
                    (unsigned long long)memcpy->end,
                    (unsigned long long)(memcpy->end - memcpy->start),
                    (unsigned long long)memcpy->bytes,
                    GetMemoryKindString(memcpy->srcKind),
                    GetMemoryKindString(memcpy->dstKind),
                    memcpy->correlationId);
            fprintf(out, "  deviceId=%u, streamId=%u",
                    memcpy->deviceId, memcpy->streamId);
            if (memcpy->graphNodeId != 0) {
                fprintf(out, ", graphId=%u, graphNodeId=%llu",
                        memcpy->graphId, (unsigned long long)memcpy->graphNodeId);
                // Check if we have API info for this graph node
                NodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(memcpy->graphNodeId);
                if (it != nodeIdCorrelationMap.end()) {
                    fprintf(out, "\n  Graph node created by API: %s (correlationId=%u)",
                            it->second.pFunctionName, it->second.correlationId);
                }
            }
            fprintf(out, "\n");
            break;
        }

        case CUPTI_ACTIVITY_KIND_MEMSET: {
            CUpti_ActivityMemset4 *memset = (CUpti_ActivityMemset4 *)record;
            fprintf(out, "%s [%llu, %llu] duration=%llu ns, size=%llu bytes, "
                    "value=%u, correlationId=%u\n",
                    GetActivityKindString(memset->kind),
                    (unsigned long long)memset->start,
                    (unsigned long long)memset->end,
                    (unsigned long long)(memset->end - memset->start),
                    (unsigned long long)memset->bytes,
                    memset->value,
                    memset->correlationId);
            fprintf(out, "  deviceId=%u, streamId=%u\n",
                    memset->deviceId, memset->streamId);
            break;
        }

        case CUPTI_ACTIVITY_KIND_MEMCPY2: {
            CUpti_ActivityMemcpyPtoP4 *memcpy2 = (CUpti_ActivityMemcpyPtoP4 *)record;
            fprintf(out, "%s \"%s\" [%llu, %llu] duration=%llu ns, size=%llu bytes, "
                    "srcKind=%s, dstKind=%s, correlationId=%u\n",
                    GetActivityKindString(memcpy2->kind),
                    GetMemcpyKindString(memcpy2->copyKind),
                    (unsigned long long)memcpy2->start,
                    (unsigned long long)memcpy2->end,
                    (unsigned long long)(memcpy2->end - memcpy2->start),
                    (unsigned long long)memcpy2->bytes,
                    GetMemoryKindString(memcpy2->srcKind),
                    GetMemoryKindString(memcpy2->dstKind),
                    memcpy2->correlationId);
            fprintf(out, "  deviceId=%u, streamId=%u, srcDeviceId=%u, dstDeviceId=%u\n",
                    memcpy2->deviceId, memcpy2->streamId,
                    memcpy2->srcDeviceId, memcpy2->dstDeviceId);
            break;
        }

        case CUPTI_ACTIVITY_KIND_MEMORY2: {
            CUpti_ActivityMemory4 *memory2 = (CUpti_ActivityMemory4 *)record;
            fprintf(out, "%s [%llu] memoryOperation=%s, memoryKind=%s, size=%llu bytes, "
                    "address=0x%llx, correlationId=%u\n",
                    GetActivityKindString(memory2->kind),
                    (unsigned long long)memory2->timestamp,
                    GetMemoryOperationTypeString(memory2->memoryOperationType),
                    GetMemoryKindString(memory2->memoryKind),
                    (unsigned long long)memory2->bytes,
                    (unsigned long long)memory2->address,
                    memory2->correlationId);
            fprintf(out, "  deviceId=%u, contextId=%u, streamId=%u, processId=%u, isAsync=%u\n",
                    memory2->deviceId, memory2->contextId, memory2->streamId,
                    memory2->processId, memory2->isAsync);
            fprintf(out, "  memoryPool=%s, poolAddress=0x%llx, poolThreshold=%llu\n",
                    GetMemoryPoolTypeString(memory2->memoryPoolConfig.memoryPoolType),
                    (unsigned long long)memory2->memoryPoolConfig.address,
                    (unsigned long long)memory2->memoryPoolConfig.releaseThreshold);
            if (memory2->memoryPoolConfig.memoryPoolType == CUPTI_ACTIVITY_MEMORY_POOL_TYPE_LOCAL) {
                fprintf(out, "  poolSize=%llu, poolUtilizedSize=%llu\n",
                        (unsigned long long)memory2->memoryPoolConfig.pool.size,
                        (unsigned long long)memory2->memoryPoolConfig.utilizedSize);
            }
            break;
        }

        case CUPTI_ACTIVITY_KIND_RUNTIME:
        case CUPTI_ACTIVITY_KIND_DRIVER: {
            CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
            const char *name = NULL;
            if (api->kind == CUPTI_ACTIVITY_KIND_RUNTIME) {
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_RUNTIME_API, api->cbid, &name);
            } else {
                cuptiGetCallbackName(CUPTI_CB_DOMAIN_DRIVER_API, api->cbid, &name);
            }
            fprintf(out, "%s [%llu, %llu] duration=%llu ns, \"%s\", "
                    "correlationId=%u, processId=%u, threadId=%u\n",
                    GetActivityKindString(api->kind),
                    (unsigned long long)api->start,
                    (unsigned long long)api->end,
                    (unsigned long long)(api->end - api->start),
                    name ? name : "<unknown>",
                    api->correlationId,
                    api->processId,
                    api->threadId);
            break;
        }

        default:
            fprintf(out, "%s (kind=%d) - detailed info not implemented\n",
                    GetActivityKindString(record->kind), record->kind);
            break;
    }

    fflush(out);
}

// ============================================================================
// CUPTI Buffer Management
// ============================================================================

static void CUPTIAPI BufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *buffer = (uint8_t *)malloc(BUF_SIZE);
    if (*buffer == NULL) {
        fprintf(stderr, "Error: Failed to allocate buffer\n");
        exit(EXIT_FAILURE);
    }
    *size = BUF_SIZE;
    *maxNumRecords = 0;
}

static void CUPTIAPI BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                                      size_t size, size_t validSize) {
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                PrintActivity(record);
            } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED) {
                break;
            } else {
                CUPTI_CALL(status);
            }
        } while (1);

        // Report any dropped records
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            fprintf(g_state.outputFile, "Warning: Dropped %llu activity records\n",
                    (unsigned long long)dropped);
        }
    }

    free(buffer);
}

// ============================================================================
// Activity Management
// ============================================================================

static void EnableActivities() {
    // Always enable runtime and kernel tracking
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));

    printf("Enabled: RUNTIME, CONCURRENT_KERNEL\n");

    // Optional: Driver API
    const char *enableDriver = getenv("CUPTI_ENABLE_DRIVER");
    if (enableDriver && atoi(enableDriver) == 1) {
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
        printf("Enabled: DRIVER, OVERHEAD\n");
    }

    // Optional: Memory operations
    const char *enableMemory = getenv("CUPTI_ENABLE_MEMORY");
    if (enableMemory && atoi(enableMemory) == 1) {
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMORY2));
        printf("Enabled: MEMCPY, MEMSET, MEMCPY2, MEMORY2\n");
    }
}

static void DisableActivities() {
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2);
    cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMORY2);
}

// ============================================================================
// Exit Handler
// ============================================================================

static void AtExitHandler() {
    if (g_state.tracingEnabled) {
        printf("Flushing CUPTI activity buffers...\n");
        DisableActivities();
        cuptiActivityFlushAll(1);  // Force flush
    }

    if (g_state.outputFile && g_state.outputFile != stdout && g_state.outputFile != stderr) {
        fflush(g_state.outputFile);
        fclose(g_state.outputFile);
        g_state.outputFile = NULL;
    }

    printf("CUPTI trace injection shutdown complete.\n");
}

// ============================================================================
// CUPTI Callbacks
// ============================================================================

static void CUPTIAPI CallbackHandler(void *userdata, CUpti_CallbackDomain domain,
                                      CUpti_CallbackId cbid, void *cbdata) {
    static const char *s_pFunctionName = NULL;
    static uint32_t s_correlationId = 0;

    const CUpti_CallbackData *callbackData = (CUpti_CallbackData *)cbdata;

    // Clear any previous errors
    cuptiGetLastError();

    switch (domain) {
        case CUPTI_CB_DOMAIN_RUNTIME_API: {
            // Store API name and correlation ID for graph node tracking
            if (callbackData->callbackSite == CUPTI_API_ENTER) {
                s_correlationId = callbackData->correlationId;
                s_pFunctionName = callbackData->functionName;
            }

            // Flush on device reset
            if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020) {
                if (callbackData->callbackSite == CUPTI_API_ENTER) {
                    cuptiActivityFlushAll(0);
                }
            }
            break;
        }

        case CUPTI_CB_DOMAIN_RESOURCE: {
            CUpti_ResourceData *resourceData = (CUpti_ResourceData *)cbdata;

            if (cbid == CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED) {
                // Don't store info for nodes created during graph instantiate
                if (s_pFunctionName && strncmp(s_pFunctionName, "cudaGraphInstantiate",
                                                strlen("cudaGraphInstantiate")) == 0) {
                    break;
                }

                CUpti_GraphData *graphData = (CUpti_GraphData *)resourceData->resourceDescriptor;
                uint64_t nodeId;

                // Query and store graph node ID with API info
                if (cuptiGetGraphNodeId(graphData->node, &nodeId) == CUPTI_SUCCESS) {
                    ApiData apiData;
                    apiData.correlationId = s_correlationId;
                    apiData.pFunctionName = s_pFunctionName;
                    nodeIdCorrelationMap[nodeId] = apiData;
                }
            }
            else if (cbid == CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED) {
                CUpti_GraphData *graphData = (CUpti_GraphData *)resourceData->resourceDescriptor;
                uint64_t nodeId, originalNodeId;

                // Update map with cloned node ID
                if (cuptiGetGraphNodeId(graphData->originalNode, &originalNodeId) == CUPTI_SUCCESS) {
                    NodeIdApiDataMap::iterator it = nodeIdCorrelationMap.find(originalNodeId);
                    if (it != nodeIdCorrelationMap.end()) {
                        if (cuptiGetGraphNodeId(graphData->node, &nodeId) == CUPTI_SUCCESS) {
                            ApiData apiData = it->second;
                            nodeIdCorrelationMap.erase(it);
                            nodeIdCorrelationMap[nodeId] = apiData;
                        }
                    }
                }
            }
            break;
        }

        default:
            break;
    }
}

// ============================================================================
// Initialization
// ============================================================================

static void SetupCupti() {
    // Configure output file
    const char *outputPath = getenv("CUPTI_TRACE_OUTPUT_FILE");
    if (!outputPath) {
        outputPath = "stdout";
    }

    if (strcmp(outputPath, "stdout") == 0) {
        g_state.outputFile = stdout;
        printf("CUPTI trace output: stdout\n");
    } else {
        g_state.outputFile = fopen(outputPath, "w");
        if (!g_state.outputFile) {
            fprintf(stderr, "Failed to open '%s', using stdout\n", outputPath);
            g_state.outputFile = stdout;
        } else {
            printf("CUPTI trace output: %s\n", outputPath);
        }
    }

    // Subscribe to CUPTI callbacks
    CUPTI_CALL(cuptiSubscribe(&g_state.subscriberHandle,
                              (CUpti_CallbackFunc)CallbackHandler, NULL));

    // Enable callback for cudaDeviceReset
    CUPTI_CALL(cuptiEnableCallback(1, g_state.subscriberHandle,
                                   CUPTI_CB_DOMAIN_RUNTIME_API,
                                   CUPTI_RUNTIME_TRACE_CBID_cudaDeviceReset_v3020));

    // Enable callbacks for CUDA graph node tracking
    CUPTI_CALL(cuptiEnableCallback(1, g_state.subscriberHandle,
                                   CUPTI_CB_DOMAIN_RESOURCE,
                                   CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED));
    CUPTI_CALL(cuptiEnableCallback(1, g_state.subscriberHandle,
                                   CUPTI_CB_DOMAIN_RESOURCE,
                                   CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));

    // Enable all runtime API callbacks to track function names
    CUPTI_CALL(cuptiEnableDomain(1, g_state.subscriberHandle,
                                 CUPTI_CB_DOMAIN_RUNTIME_API));

    // Register buffer callbacks
    CUPTI_CALL(cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted));

    // Enable activities
    EnableActivities();
}

// ============================================================================
// Injection Entry Point
// ============================================================================

#ifdef _WIN32
extern "C" __declspec(dllexport) int InitializeInjection(void)
#else
extern "C" int InitializeInjection(void)
#endif
{
    if (g_state.initialized) {
        return 1;
    }

    g_state.initMutex.lock();

    if (g_state.initialized) {
        g_state.initMutex.unlock();
        return 1;
    }

    printf("=== CUPTI Trace Injection (Simplified) ===\n");

    // Register exit handler
    atexit(AtExitHandler);

    // Setup CUPTI
    SetupCupti();

    g_state.tracingEnabled = 1;
    g_state.initialized = 1;

    printf("CUPTI trace injection initialized successfully.\n");
    printf("==========================================\n");

    g_state.initMutex.unlock();

    return 1;
}
