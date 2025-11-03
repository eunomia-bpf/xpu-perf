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

// Local headers
#include "cupti_trace_print.h"

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

// Graph node tracking map (declared in cupti_trace_print.h)
NodeIdApiDataMap nodeIdCorrelationMap;

struct GlobalState {
    CUpti_SubscriberHandle subscriberHandle;
    FILE *outputFile;
    int tracingEnabled;
    std::mutex initMutex;
    int initialized;
};

static GlobalState g_state = {0};

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
                PrintActivity(record, g_state.outputFile);
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

    // Optional: Enable CUDA graph node tracking via CUPTI_ENABLE_GRAPH=1
    const char *enableGraph = getenv("CUPTI_ENABLE_GRAPH");
    if (enableGraph && atoi(enableGraph) == 1) {
        CUPTI_CALL(cuptiEnableCallback(1, g_state.subscriberHandle,
                                       CUPTI_CB_DOMAIN_RESOURCE,
                                       CUPTI_CBID_RESOURCE_GRAPHNODE_CREATED));
        CUPTI_CALL(cuptiEnableCallback(1, g_state.subscriberHandle,
                                       CUPTI_CB_DOMAIN_RESOURCE,
                                       CUPTI_CBID_RESOURCE_GRAPHNODE_CLONED));
        // Enable all runtime API callbacks to track function names for graph nodes
        CUPTI_CALL(cuptiEnableDomain(1, g_state.subscriberHandle,
                                     CUPTI_CB_DOMAIN_RUNTIME_API));
        printf("CUDA graph tracking: ENABLED\n");
    } else {
        printf("CUDA graph tracking: DISABLED (set CUPTI_ENABLE_GRAPH=1 to enable)\n");
    }

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
