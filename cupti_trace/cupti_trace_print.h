/*
 * CUPTI Activity Record Printing Functions
 *
 * This header contains all the printing and formatting functions for CUPTI
 * activity records. Supports CUDA graphs, memory operations, kernels, and APIs.
 */

#ifndef CUPTI_TRACE_PRINT_H_
#define CUPTI_TRACE_PRINT_H_

#include <stdio.h>
#include <cupti.h>
#include <map>

// Forward declaration for graph node tracking
typedef struct ApiData_st {
    const char *pFunctionName;
    uint32_t correlationId;
} ApiData;

typedef std::map<uint64_t, ApiData> NodeIdApiDataMap;
extern NodeIdApiDataMap nodeIdCorrelationMap;

// ============================================================================
// String Conversion Functions
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
// Activity Record Printing Functions
// ============================================================================

static void PrintActivity(CUpti_Activity *record, FILE *out) {
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

#endif // CUPTI_TRACE_PRINT_H_
