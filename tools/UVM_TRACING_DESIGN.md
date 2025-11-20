# UVM 决策追踪 Bpftrace 脚本设计文档

## 1. 概述

基于 NVIDIA UVM 文档分析和可追踪函数列表，设计一系列 bpftrace 脚本来追踪 UVM 的关键决策过程，包括：
- LRU 页面替换决策
- Prefetch 预取策略
- Thrashing 检测和缓解
- 页面迁移决策
- 内存驱逐 (Eviction) 流程

## 2. 现有脚本分析

### 已有脚本 (function-script 目录)
1. `trace_uvm_va_block_select_residency.bt` - 追踪页面驻留位置选择
2. `trace_uvm_perf_prefetch_get_hint.bt` - 追踪预取提示
3. `trace_uvm_perf_thrashing_get_hint_simple.bt` - 追踪 thrashing 提示

### 缺失的追踪点
根据文档分析，以下关键决策点尚未覆盖：
- LRU 驱逐选择和更新
- Prefetch 区域计算细节
- Thrashing 缓解策略 (Pin/Throttle)
- 页面迁移触发原因
- 内存分配失败后的驱逐流程

## 3. 新脚本设计

### 3.1 LRU 驱逐决策追踪

**脚本名称**: `trace_uvm_lru_eviction.bt`

**追踪目标**:
- 驱逐触发时机
- LRU 选择逻辑
- 驱逐优先级 (Free → Unused → LRU)
- 被驱逐的 chunk 信息

**关键函数**:
```
pick_root_chunk_to_evict          # 选择驱逐目标 (Line 1460 in uvm_pmm_gpu.c)
chunk_update_lists_locked         # 更新 LRU 列表 (Line 627)
evict_root_chunk                  # 执行驱逐 (Line 300)
uvm_pmm_gpu_mark_chunk_evicted    # 标记已驱逐 (Line 2602)
```

**追踪信息**:
1. 驱逐原因 (内存不足)
2. 候选 chunk 来源 (free_list / unused / LRU)
3. 被驱逐 chunk 的物理地址
4. 驱逐前后的内存统计
5. 驱逐耗时

**输出格式**:
```
[EVICTION] Triggered by alloc failure
  → Candidate source: LRU (va_block_used)
  → Chunk PA: 0x12345000, Size: 2MB
  → Evicted pages: 512
  → Duration: 1234 us
```

---

### 3.2 Chunk 更新和 LRU 维护追踪

**脚本名称**: `trace_uvm_chunk_lifecycle.bt`

**追踪目标**:
- Chunk 分配和释放
- LRU 列表更新时机
- Chunk 状态转换 (FREE → ALLOCATED → TEMP_PINNED → EVICTED)

**关键函数**:
```
chunk_update_lists_locked         # LRU 更新 (Line 627)
uvm_pmm_gpu_alloc_user           # 用户内存分配 (Line 2593)
uvm_pmm_gpu_free                 # 释放内存 (Line 2599)
uvm_pmm_gpu_unpin_allocated      # Unpin 后更新 LRU (Line 2614)
root_chunk_update_eviction_list  # 更新驱逐列表 (Line 1669)
```

**追踪信息**:
1. Chunk 地址和大小
2. 状态转换时间戳
3. LRU 位置变化 (头部 → 尾部)
4. Pin/Unpin 事件
5. 分配标志 (是否允许驱逐)

**输出格式**:
```
[CHUNK] Alloc: PA=0x12345000, Size=2MB, Flags=EVICT
  → State: FREE → ALLOCATED
  → LRU: Added to tail (MRU position)

[CHUNK] Unpin: PA=0x12345000
  → State: TEMP_PINNED → ALLOCATED
  → LRU: Moved to tail (refreshed MRU)
```

---

### 3.3 Prefetch 决策详细追踪

**脚本名称**: `trace_uvm_prefetch_decision.bt`

**追踪目标**:
- Prefetch 触发条件
- Bitmap tree 遍历过程
- 阈值判断 (51% occupancy)
- 预取区域大小

**关键函数**:
```
uvm_perf_prefetch_get_hint_va_block    # 顶层接口 (Line 2564)
compute_prefetch_mask                  # 计算预取掩码 (Line 246)
uvm_perf_prefetch_bitmap_tree_iter_init      # 树遍历初始化 (Line 2561)
uvm_perf_prefetch_bitmap_tree_iter_get_count # 获取计数 (Line 2559)
uvm_perf_prefetch_bitmap_tree_iter_get_range # 获取范围 (Line 2560)
grow_fault_granularity_if_no_thrashing      # 增长粒度 (Line 360)
```

**追踪信息**:
1. Fault page 索引
2. Bitmap tree 层数和叶子数
3. 每层的 occupancy 计算
4. 阈值比较结果
5. 最终预取区域 (start, end)
6. 预取页面数量
7. 是否受 thrashing 影响

**输出格式**:
```
[PREFETCH] Fault at page index 128
  → Tree: 3 levels, 512 leaves
  → L0 (leaf): counter=1/1 (100%) ✓ → region=[128, 129)
  → L1: counter=150/128 (117%) ✓ → region=[0, 128)
  → L2: counter=200/256 (78%) ✓ → region=[0, 256)
  → Root: counter=256/512 (50%) ✗
  → Final prefetch region: [0, 256) = 256 pages
  → Thrashing pages excluded: 10
```

---

### 3.4 Thrashing 检测和缓解追踪

**脚本名称**: `trace_uvm_thrashing_mitigation.bt`

**追踪目标**:
- Thrashing 检测逻辑
- Pin/Throttle 决策
- Thrashing 页面统计
- 缓解效果

**关键函数**:
```
uvm_perf_thrashing_get_hint           # 获取 thrashing 提示 (Line 2570)
thrashing_event_cb                    # Thrashing 事件回调 (Line 1784)
thrashing_throttle_end_processor.isra.0  # 节流处理 (Line 1787)
thrashing_unpin_pages                 # 解除 pin (Line 1788)
uvm_perf_thrashing_get_thrashing_pages    # 获取 thrashing 页面 (Line 2571)
uvm_perf_thrashing_get_thrashing_processors # 获取 thrashing 处理器 (Line 2572)
```

**追踪信息**:
1. Thrashing 检测触发
2. 涉及的处理器 (CPU, GPU)
3. Thrashing 页面数量
4. 迁移频率统计
5. 缓解策略类型 (PIN / THROTTLE / NONE)
6. Throttle 时长
7. Pin 页面数量

**输出格式**:
```
[THRASHING] Detected: VA range [0x1000-0x2000]
  → Processors: CPU ↔ GPU0
  → Thrashing pages: 64
  → Migration frequency: 10 times/sec
  → Mitigation: PIN (prevent migration)
  → Pinned pages: 64

[THRASHING] Throttle: VA range [0x3000-0x4000]
  → Processors: GPU0 ↔ GPU1
  → Throttle duration: 100 ms
  → Reason: Cross-GPU thrashing
```

---

### 3.5 页面迁移决策追踪

**脚本名称**: `trace_uvm_migration_decision.bt`

**追踪目标**:
- 迁移触发原因 (page fault / prefetch / access counters)
- 源和目标处理器
- 迁移页面数量
- 迁移耗时

**关键函数**:
```
uvm_va_block_add_mappings_after_migration  # 迁移后映射 (Line 2982)
block_copy_resident_pages                  # 复制驻留页面 (Line 148)
on_block_migration_complete                # 迁移完成 (Line 1439)
record_migration_events                    # 记录迁移事件 (Line 1655)
uvm_tools_record_migration                 # 工具记录迁移 (Line 2932)
migration_should_do_cpu_preunmap          # CPU 预解映射 (Line 552)
```

**追踪信息**:
1. 迁移原因 (FAULT / PREFETCH / ACCESS_COUNTER / EVICTION)
2. 源处理器 ID
3. 目标处理器 ID
4. 迁移页面数量和地址范围
5. 是否是首次访问 (first-touch)
6. 迁移数据量 (bytes)
7. 迁移耗时 (us)
8. 是否触发预取

**输出格式**:
```
[MIGRATION] Reason: PAGE_FAULT
  → Source: CPU (ID=0)
  → Destination: GPU0 (ID=1)
  → VA range: [0x10000-0x11000]
  → Pages: 16 (4KB each)
  → Data transferred: 64 KB
  → Duration: 234 us
  → Prefetch triggered: Yes (128 additional pages)
```

---

### 3.6 内存分配和驱逐流程追踪

**脚本名称**: `trace_uvm_alloc_evict_flow.bt`

**追踪目标**:
- 内存分配请求
- 分配失败和重试
- 驱逐触发
- 完整的 alloc → evict → retry 流程

**关键函数**:
```
uvm_pmm_gpu_alloc_user               # 用户内存分配 (Line 2593)
pick_and_evict_root_chunk            # 选择并驱逐 (Line 1617)
find_and_retain_va_block_to_evict    # 查找 VA block 驱逐 (Line 318)
uvm_va_block_evict_chunks            # 驱逐 chunks (Line 3001)
block_add_eviction_mappings          # 添加驱逐映射 (Line 140)
```

**追踪信息**:
1. 分配请求大小
2. 分配标志 (NONE / EVICT / ZERO)
3. 第一次分配尝试结果
4. 驱逐触发原因 (NV_ERR_NO_MEMORY)
5. 驱逐的 chunk 信息
6. 第二次分配尝试结果
7. 总耗时

**输出格式**:
```
[ALLOC] Request: size=2MB, flags=NONE
  → Result: NV_ERR_NO_MEMORY
  → Free memory: 0 MB

[EVICT] Triggered by allocation failure
  → Evicting chunk: PA=0x50000000, size=2MB
  → Eviction duration: 456 us

[ALLOC] Retry: size=2MB, flags=EVICT
  → Result: SUCCESS
  → Allocated chunk: PA=0x50000000 (reused)
  → Total duration: 678 us
```

---

### 3.7 策略参数调优追踪

**脚本名称**: `trace_uvm_policy_tuning.bt`

**追踪目标**:
- 当前策略参数值
- 策略更改事件
- 参数对决策的影响

**关键函数**:
```
uvm_test_set_page_prefetch_policy      # 设置预取策略 (Line 2871)
uvm_test_set_page_thrashing_policy     # 设置 thrashing 策略 (Line 2872)
uvm_test_set_prefetch_faults_reenable_lapse  # 设置预取 fault 重启延迟 (Line 2873)
uvm_va_policy_set_preferred_location   # 设置首选位置 (Line 3095)
uvm_va_policy_set_range                # 设置策略范围 (Line 3096)
```

**追踪信息**:
1. 当前 prefetch threshold 值
2. Thrashing 检测阈值
3. Preferred location 变更
4. Read duplication 策略
5. Access counter 配置

**输出格式**:
```
[POLICY] Prefetch threshold: 51%
[POLICY] Thrashing detection: ENABLED
[POLICY] Preferred location: GPU0
[POLICY] Read duplication: ENABLED for VA range [0x0-0x100000]
```

---

## 4. 实现计划

### Phase 1: 核心决策追踪 (优先级: 高)
- [x] ~~trace_uvm_va_block_select_residency.bt~~ (已存在)
- [ ] `trace_uvm_lru_eviction.bt` - LRU 驱逐决策
- [ ] `trace_uvm_migration_decision.bt` - 迁移决策

### Phase 2: 性能优化追踪 (优先级: 高)
- [x] ~~trace_uvm_perf_prefetch_get_hint.bt~~ (已存在，需增强)
- [ ] `trace_uvm_prefetch_decision.bt` - 详细预取决策
- [x] ~~trace_uvm_perf_thrashing_get_hint_simple.bt~~ (已存在，需增强)
- [ ] `trace_uvm_thrashing_mitigation.bt` - Thrashing 缓解

### Phase 3: 内存管理追踪 (优先级: 中)
- [ ] `trace_uvm_chunk_lifecycle.bt` - Chunk 生命周期
- [ ] `trace_uvm_alloc_evict_flow.bt` - 分配驱逐流程

### Phase 4: 策略分析 (优先级: 低)
- [ ] `trace_uvm_policy_tuning.bt` - 策略参数
- [ ] `trace_uvm_access_counters.bt` - Access counter 决策

---

## 5. 技术挑战和解决方案

### 5.1 挑战：部分函数是内联或静态函数
**问题**: 如 `compute_prefetch_region()` 可能被内联优化
**解决方案**:
1. 追踪调用链上层函数 (如 `uvm_perf_prefetch_get_hint_va_block`)
2. 使用 USDT tracepoint (如果可用)
3. 使用 kprobe 的 return probe 获取返回值

### 5.2 挑战：需要访问内核数据结构
**问题**: Bpftrace 无法直接访问复杂的内核结构体
**解决方案**:
1. 追踪函数参数和返回值
2. 使用已有的 tracepoint
3. 结合 dmesg 和内核日志分析

### 5.3 挑战：高频事件导致性能开销
**问题**: 如 `chunk_update_lists_locked` 调用频繁
**解决方案**:
1. 使用采样 (每 N 次事件记录一次)
2. 添加过滤条件 (如只追踪特定 GPU)
3. 使用聚合统计而非逐条记录

### 5.4 挑战：函数名包含 `.isra.0` 等后缀
**问题**: 编译器优化导致函数名变化
**解决方案**:
1. 使用通配符: `kprobe:thrashing_throttle_end_processor*`
2. 先用 `grep` 确认实际函数名
3. 提供多个备选函数名

---

## 6. 测试计划

### 6.1 测试环境
- NVIDIA GPU with UVM support
- CUDA 应用程序 (如 vectorAdd)
- 内存压力测试程序 (分配超过 GPU 内存的 Managed Memory)

### 6.2 测试场景

#### 场景 1: LRU 驱逐测试
```bash
# 分配 8GB Managed Memory (超过 GPU 4GB 内存)
./test_oversubscription --size=8G
```
**预期追踪结果**:
- 观察到 `pick_root_chunk_to_evict` 被调用
- 看到从 LRU 列表选择 chunk
- 记录驱逐时间和频率

#### 场景 2: Prefetch 测试
```bash
# 顺序访问大块内存
./test_sequential_access --size=1G --stride=4K
```
**预期追踪结果**:
- 首次访问触发 page fault
- Prefetch 算法预取相邻页面
- 观察 bitmap tree 遍历过程

#### 场景 3: Thrashing 测试
```bash
# CPU 和 GPU 交替访问同一内存
./test_thrashing --pattern=ping-pong
```
**预期追踪结果**:
- 检测到频繁的 CPU ↔ GPU 迁移
- Thrashing 缓解策略触发 (Pin 或 Throttle)
- 迁移频率下降

### 6.3 验证方法
1. 对比 bpftrace 输出和 `nvidia-smi` 统计
2. 检查 `/proc` 文件系统中的 UVM 统计信息
3. 使用 `nvprof` 或 `nsys` 验证事件时序

---

## 7. 脚本命名规范

```
trace_uvm_<subsystem>_<functionality>.bt

subsystem:
  - lru         : LRU 相关
  - prefetch    : 预取相关
  - thrashing   : Thrashing 相关
  - migration   : 迁移相关
  - chunk       : Chunk 管理
  - alloc       : 内存分配
  - policy      : 策略管理

functionality:
  - decision    : 决策过程
  - lifecycle   : 生命周期
  - mitigation  : 缓解措施
  - flow        : 完整流程
  - tuning      : 参数调优
```

---

## 8. 输出格式规范

### 8.1 时间戳格式
```
[HH:MM:SS.mmm] [CATEGORY] Message
```

### 8.2 分类标签
- `[EVICTION]` - 驱逐事件
- `[PREFETCH]` - 预取事件
- `[THRASHING]` - Thrashing 事件
- `[MIGRATION]` - 迁移事件
- `[ALLOC]` - 分配事件
- `[POLICY]` - 策略变更
- `[CHUNK]` - Chunk 操作

### 8.3 地址格式
- 物理地址: `PA=0x12345000`
- 虚拟地址: `VA=0x7fff12345000`
- 页面索引: `page_index=128`

### 8.4 统计输出
```
BEGIN {
    printf("Tracing UVM ... Hit Ctrl-C to end.\n");
}

END {
    printf("\n=== Statistics ===\n");
    printf("Total evictions: %d\n", @eviction_count);
    printf("Avg eviction time: %d us\n", @eviction_time_sum / @eviction_count);
}
```

---

## 9. 参考文档

### 内部文档
- `UVM_LRU_POLICY.md` - LRU 替换策略分析
- `UVM_PREFETCH_AND_POLICY_HOOKS.md` - 预取机制和策略 hook
- `nvidia_test_results.log` - 可追踪函数测试结果
- `nvidia_traceable_functions.txt` - 所有可追踪函数列表

### 源代码位置
- `kernel-open/nvidia-uvm/uvm_pmm_gpu.c` - PMM 实现
- `kernel-open/nvidia-uvm/uvm_perf_prefetch.c` - 预取实现
- `kernel-open/nvidia-uvm/uvm_perf_thrashing.c` - Thrashing 检测
- `kernel-open/nvidia-uvm/uvm_va_block.c` - VA block 管理

### 关键行号索引
参见 `UVM_LRU_POLICY.md` 第 314-323 行的"关键代码位置索引"表格

---

## 10. 下一步行动

1. **立即开始**: 实现 Phase 1 的 LRU 驱逐追踪脚本
2. **增强现有脚本**: 为已存在的脚本添加更详细的输出
3. **建立测试套件**: 创建自动化测试脚本验证追踪准确性
4. **文档化发现**: 记录追踪过程中发现的 UVM 行为模式

---

**文档版本**: v1.0
**创建时间**: 2025-11-19
**作者**: Based on UVM analysis and traceable functions
**状态**: 设计完成，待实现
