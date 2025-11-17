# NVIDIA Kernel Function Tracing Test Results

Test Application: `/home/yunwei37/workspace/xpu-perf/test/mock-app/vectorAdd`

## Summary

Total NVIDIA traceable functions found: **3,202**
- nvidia: 885 functions
- nvidia_uvm: 2,019 functions
- nvidia_drm: 192 functions
- nvidia_modeset: 106 functions

## Individual Wildcard Tests

### 1. __nv_drm_* Functions (42 probes)

**Status**: No activity detected
- Attached successfully
- No functions called during vectorAdd execution
- These are DRM/display-related, not used by CUDA compute workloads

### 2. nv_* Functions (533 probes)

**Top Function Calls**:
```
nv_get_ctl_state: 104,417 calls
nv_post_event: 35,355 calls
nv_get_kern_phys_address: 15,892 calls
nv_uvm_event_interrupt: 3,094 calls
nv_printf: 785 calls
```

**Memory Operations**:
- Page allocation/deallocation: ~28 calls each
- DMA mapping: 32 map/unmap operations
- User/kernel mappings: ~22-23 calls each

**Device Operations**:
- Open/close: 14 calls each (per GPU context)
- File private management: 36 get/put, 39 free calls

### 3. nvidia_* Functions (59 probes)

**Top Function Calls**:
```
nvidia_poll: 49,714 calls (most frequent!)
nvidia_isr: 2,856 calls
nvidia_isr_msix: 2,713 calls
nvidia_unlocked_ioctl: 333 calls
nvidia_open/close: 39 calls each
nvidia_mmap: 26 calls
```

**Key Findings**:
- Heavy polling activity (49k calls)
- Interrupt handling very active (5.5k ISR calls total)
- 26 mmap operations for memory mapping
- 1 RC (Runlist Controller) timer callback

### 4. uvm_* Functions (1,301 probes)

**Top Function Calls**:
```
uvm_range_tree_init: 6,915 calls
uvm_rm_mem_get_cpu_va: 6,723 calls
uvm_tracker_remove_completed: 6,253 calls
uvm_spin_loop: 5,632 calls
uvm_push_inline_data_get: 4,777 calls
uvm_pte_batch_write_pte: 1,025 calls
```

**Channel Operations**:
- Begin/end push: 202 calls each
- Pushbuffer operations: 202-272 calls
- Channel creation: 10 channels
- Channel stop/destroy/detach: 16 calls each

**Memory Management**:
- VA range operations: ~40 create, ~40 destroy
- External mappings: 27 operations
- TLB invalidation: 80 batch operations

**GPU Management**:
- GPU registration/lookup: ~50 calls
- VA space operations: comprehensive lifecycle
- Tracker/semaphore management: 340+ operations

### 5. nvKms* Functions (10 probes)

**Status**: No activity detected
- KMS (Kernel Mode Setting) for display
- Not used in CUDA compute workloads

### 6. nvUvm* Functions (79 probes)

**Top Function Calls**:
```
nvUvmInterfaceHasPendingNonReplayableFaults: 2,836 calls
nvUvmGetSafeStack: 61 calls
nvUvmInterfaceDupMemory: 27 calls
nvUvmInterfaceGetExternalAllocPtes: 27 calls
nvUvmInterfaceFreeDupedHandle: 27 calls
```

**Channel Operations**:
- Retain/Release/Stop/Bind: 16 calls each
- GetChannelResourcePtes: 10 calls

**Address Space Operations**:
- DupAddressSpace: 1
- SetPageDirectory: 1
- UnsetPageDirectory: 1
- AddressSpaceDestroy: 1

## Key Insights

1. **Most Active Functions**:
   - `nvidia_poll`: 49,714 calls (polling for GPU completion)
   - `nv_get_ctl_state`: 104,417 calls (control state queries)
   - `nv_post_event`: 35,355 calls (event posting)

2. **Interrupt Handling**:
   - 5,569 total ISR calls (nvidia_isr + nvidia_isr_msix)
   - 3,094 UVM event interrupts

3. **Channel/Push Operations**:
   - 202 push begin/end pairs (GPU command submission)
   - 16 user channels created and managed

4. **Memory Operations**:
   - Extensive page table management (1,025 PTE writes)
   - 27 external memory mappings
   - 32 DMA mappings

5. **No Display Activity**:
   - __nv_drm_* and nvKms* functions not called
   - Confirms CUDA compute workload has no display operations

## Files Generated

- `/tmp/test_nv_drm.log` - DRM function test results
- `/tmp/test_nv.log` - nv_* function test results
- `/tmp/test_nvidia.log` - nvidia_* function test results
- `/tmp/test_uvm_results.log` - uvm_* function test results
- `/tmp/test_nvkms.log` - nvKms* function test results
- `/tmp/test_nvuvm.log` - nvUvm* function test results

## Test Scripts

Individual test scripts created:
- `test_nv_drm.bt` - Test __nv_drm_* functions
- `test_nv.bt` - Test nv_* functions
- `test_nvidia.bt` - Test nvidia_* functions
- `test_uvm.bt` - Test uvm_* functions
- `test_nvkms.bt` - Test nvKms* functions
- `test_nvuvm.bt` - Test nvUvm* functions

All scripts successfully tested with vectorAdd CUDA application.
