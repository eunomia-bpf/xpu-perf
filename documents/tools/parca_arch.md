# Parca 项目架构与技术分析文档

> 详细的代码分析、模块架构、工作原理和依赖关系文档

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈与核心依赖](#2-技术栈与核心依赖)
3. [项目结构](#3-项目结构)
4. [核心架构设计](#4-核心架构设计)
5. [数据流与工作原理](#5-数据流与工作原理)
6. [核心模块详解](#6-核心模块详解)
7. [存储架构](#7-存储架构)
8. [API 接口设计](#8-api-接口设计)
9. [关键算法与优化](#9-关键算法与优化)
10. [配置系统](#10-配置系统)
11. [构建与部署](#11-构建与部署)
12. [性能特征](#12-性能特征)
13. [集成与扩展](#13-集成与扩展)

---

## 1. 项目概述

### 1.1 什么是 Parca

Parca 是一个**开源的持续性能剖析（Continuous Profiling）平台**，专为生产环境设计，用于分析 CPU、内存使用情况，并能精确定位到代码行级别。它通过极低的开销帮助组织节省基础设施成本、提升性能并增强系统可靠性。

**核心特性：**
- **eBPF 自动化采集器**：使用 eBPF 技术实现零侵入式性能剖析，自动发现 Kubernetes 或 systemd 环境中的目标进程
- **开放标准支持**：生产和接收 pprof 格式的性能数据，支持广泛的编程语言（C、C++、Rust、Go 等）
- **优化的存储与查询**：基于列式存储（Apache Arrow + FrostDB）高效存储原始数据，支持基于标签的多维度查询和聚合

### 1.2 应用场景

1. **成本优化**：许多组织有 20-30% 的资源被低效代码路径浪费，Parca 能快速识别优化点
2. **性能提升**：通过时间序列数据的统计分析，自信地确定热点路径进行优化
3. **故障诊断**：内存泄漏、CPU 突增、I/O 阻塞等传统难以排查的问题，通过持续性能剖析数据可轻松定位

### 1.3 设计目标

- **低准入门槛**：无需修改应用代码，部署即可使用
- **生产环境友好**：极低的性能开销（< 1% CPU）
- **可扩展性**：支持从单机到大规模集群的水平扩展
- **可观测性**：与现有监控体系（Prometheus、OpenTelemetry）无缝集成

---

## 2. 技术栈与核心依赖

### 2.1 后端技术栈

**编程语言与核心框架：**
- **Go 1.24.1**：主要开发语言
- **Apache Arrow Go v18**：列式内存数据结构
- **gRPC + grpc-gateway**：API 层（同时支持 gRPC 和 RESTful HTTP）
- **Protocol Buffers**：服务定义和数据序列化

**存储层：**
- **FrostDB**（github.com/polarsignals/frostdb）：核心列式存储引擎
  - 基于 Apache Arrow 内存表示
  - 使用 Parquet 格式持久化
  - LSM-tree 风格的压缩策略
- **BadgerDB v4**：元数据存储（KV 存储）
  - 符号化缓存
  - Debuginfo 元数据
  - 配置状态管理

**性能剖析与符号化：**
- **google/pprof**：pprof 格式解析和生成
- **go-delve/delve**：DWARF 调试信息解析
- **ianlancetaylor/demangle**：C++ 符号 demangling
- **polarsignals/iceberg-go**：Apache Iceberg 表格式支持（实验性）

**对象存储：**
- **thanos-io/objstore**：统一对象存储抽象层
  - Amazon S3
  - Google Cloud Storage
  - Azure Blob Storage
  - MinIO
  - 本地文件系统

### 2.2 前端技术栈

**框架与工具：**
- **React 18.3.1**：UI 框架
- **Next.js**：服务端渲染框架
- **TypeScript 5.8.3**：类型安全
- **Lerna 8.2.4**：Monorepo 管理
- **pnpm**：包管理器

**UI 组件与可视化：**
- **@parca/shared/components**：共享 React 组件库
- **@parca/shared/profile**：性能剖析数据可视化
- **Tailwind CSS 3.2.4**：样式框架
- **Storybook 8.6.14**：组件开发环境

### 2.3 核心依赖详解

#### 2.3.1 FrostDB - 列式存储引擎

FrostDB 是 Parca 的核心存储引擎，专为时间序列性能剖析数据设计：

```
特性：
- 列式存储：每列独立压缩，查询时只读取必要列
- 动态 Schema：支持动态标签列（pprof labels）
- 高压缩比：50-100x（RLE 字典编码 + Delta 编码 + LZ4）
- 实时查询：活跃内存缓冲区 + 持久化 Parquet 文件
- LSM-tree 压缩：多级压缩策略，自动合并小文件
```

**数据流：**
```
Arrow RecordBatch → Active Memory Buffer → WAL (可选) → Compaction → Parquet Files → Object Storage
```

#### 2.3.2 Prometheus 生态集成

- **prometheus/prometheus v0.305.0**：服务发现机制
  - Kubernetes SD
  - Consul SD
  - Static SD
  - 文件 SD
- **prometheus/client_golang**：指标导出
- **grafana/regexp**：标签匹配表达式

#### 2.3.3 OpenTelemetry 集成

- **go.opentelemetry.io/otel**：分布式追踪
- **otelgrpc**：gRPC 拦截器自动追踪
- **OTLP Profiling**（实验性）：接收 OpenTelemetry 性能剖析数据

### 2.4 构建工具链

- **buf**：Protocol Buffers 管理和生成
- **golangci-lint**：Go 代码 lint
- **GoReleaser**：发布自动化
- **Docker/Podman**：容器化构建
- **Tilt**：Kubernetes 本地开发环境

---

## 3. 项目结构

### 3.1 目录结构总览

```
parca/
├── cmd/                    # 命令行入口
│   └── parca/
│       └── main.go        # 主程序入口
├── pkg/                    # 核心业务逻辑包
│   ├── parca/             # 主服务协调器
│   ├── scrape/            # 抓取系统
│   ├── profilestore/      # 性能剖析数据存储
│   ├── ingester/          # 数据接收器
│   ├── query/             # 查询服务
│   ├── parcacol/          # FrostDB 查询器
│   ├── symbolizer/        # 符号化引擎
│   ├── debuginfo/         # 调试信息管理
│   ├── profile/           # 性能剖析数据模型
│   ├── normalizer/        # 数据格式规范化
│   ├── demangle/          # C++ 符号解码
│   ├── server/            # HTTP/gRPC 服务器
│   ├── config/            # 配置管理
│   ├── kv/                # KV 存储抽象
│   ├── cache/             # 缓存层
│   ├── hash/              # 哈希工具
│   ├── tracer/            # 分布式追踪
│   ├── telemetry/         # 遥测服务
│   └── ...
├── proto/                  # Protocol Buffers 定义
│   └── parca/
│       ├── query/         # 查询 API
│       ├── profilestore/  # 存储 API
│       ├── scrape/        # 抓取 API
│       ├── debuginfo/     # 调试信息 API
│       ├── share/         # 分享 API
│       └── telemetry/     # 遥测 API
├── gen/                    # 生成的代码
│   └── proto/go/          # 生成的 Go protobuf 代码
├── ui/                     # 前端代码
│   └── packages/
│       ├── app/           # 主应用
│       │   └── web/       # Next.js Web 应用
│       └── shared/        # 共享包
│           ├── client/    # API 客户端
│           ├── components/# React 组件
│           ├── profile/   # 性能剖析可视化
│           ├── parser/    # pprof 解析器
│           └── ...
├── deploy/                 # 部署配置
│   └── ...
├── scripts/                # 构建脚本
├── parca.yaml             # 默认配置文件
├── go.mod                 # Go 模块依赖
├── package.json           # 前端依赖（根）
└── Makefile               # 构建任务
```

### 3.2 核心包职责划分

#### 3.2.1 入口层 (cmd/)

**cmd/parca/main.go**（72 行）
- 解析命令行参数（使用 `alecthomas/kong`）
- 初始化日志系统
- 设置 Prometheus 注册表
- 调用 `parca.Run()` 启动服务

#### 3.2.2 服务协调层 (pkg/parca/)

**parca.go**（约 900 行）
- **主要功能**：
  - 初始化所有子系统（存储、抓取、查询、符号化）
  - 配置 gRPC/HTTP 服务器
  - 管理服务生命周期（使用 `oklog/run` 组）
  - 处理配置热重载

**关键代码结构**：
```go
func Run(ctx context.Context, logger log.Logger, reg *prometheus.Registry, flags *Flags, version string) error {
    // 1. 初始化追踪
    // 2. 创建存储后端（FrostDB + BadgerDB）
    // 3. 创建符号化器
    // 4. 创建查询服务
    // 5. 创建抓取管理器
    // 6. 启动 gRPC/HTTP 服务器
    // 7. 运行服务组（oklog/run）
}
```

#### 3.2.3 数据采集层 (pkg/scrape/)

**核心文件**：
- `manager.go`：抓取管理器，管理所有抓取池
- `scrape.go`：单个抓取循环实现
- `target.go`：抓取目标抽象
- `service.go`：gRPC 服务实现

**设计模式**：
```
Manager (1) → ScrapePool (N) → Target (M) → HTTP Scraper
   ↓
ServiceDiscovery (Prometheus)
```

#### 3.2.4 存储层 (pkg/profilestore/ + pkg/ingester/)

**profilestore/profilecolumnstore.go**：
- `WriteRaw()`：接收 pprof 二进制数据
- `Write()`：双向流式 Arrow 数据传输
- `ExportOTLP()`：OpenTelemetry 性能剖析数据导出

**ingester/ingester.go**：
- 简单的 FrostDB 表插入包装器
- 线程安全的字典初始化

#### 3.2.5 查询层 (pkg/query/ + pkg/parcacol/)

**pkg/query/**（约 100KB 代码）：
- `columnquery.go`：查询服务主入口
- `flamegraph_arrow.go`：火焰图生成（Arrow 格式）
- `flamegraph_table.go`：火焰图（嵌套表格式）
- `table.go`：表格视图
- `top.go`：Top N 函数排名
- `callgraph.go`：调用图
- `pprof.go`：pprof 格式导出

**pkg/parcacol/querier.go**（约 1200 行）：
- FrostDB 查询构建器
- 符号化协调
- 标签发现
- 时间范围查询

#### 3.2.6 符号化层 (pkg/symbolizer/ + pkg/debuginfo/)

**symbolizer/**：
- 地址解析为函数名/文件名/行号
- 多后端支持（DWARF、pclntab、symtab、外部 addr2line）
- 地址规范化（PIE 可执行文件支持）
- 符号化结果缓存

**debuginfo/**：
- 调试信息上传管理（三阶段：ShouldInitiate → Initiate → MarkFinished）
- Debuginfod 客户端（fallback 到上游服务器）
- 元数据存储（Build ID → 调试信息映射）

### 3.3 Proto API 定义

**proto/parca/** 目录结构：
```
parca/
├── query/v1alpha1/
│   └── query.proto          # 查询 API（Query, QueryRange, Labels, Values）
├── profilestore/v1alpha1/
│   └── profilestore.proto   # 存储 API（WriteRaw, Write, Agents）
├── scrape/v1alpha1/
│   └── scrape.proto         # 抓取 API（Targets）
├── debuginfo/v1alpha1/
│   └── debuginfo.proto      # 调试信息 API（Upload, ShouldInitiate）
├── share/v1alpha1/
│   └── share.proto          # 分享 API（ShareProfile）
└── telemetry/v1alpha1/
    └── telemetry.proto      # 遥测 API（内部使用）
```

**API 设计特点**：
- 使用 `google.api.http` 注解自动生成 RESTful 端点（grpc-gateway）
- 流式传输支持（`stream` 关键字）
- Protobuf v3 语法
- 使用 `vtprotobuf` 优化序列化性能

---

## 4. 核心架构设计

### 4.1 整体架构图

```
┌────────────────────────────────────────────────────────────────────────┐
│                          Parca Server                                   │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  HTTP/gRPC Server Layer (Port 7070)                              │  │
│  │  - grpc-gateway (gRPC → HTTP REST)                               │  │
│  │  - CORS 支持                                                      │  │
│  │  - TLS 终止                                                       │  │
│  │  - 路径前缀支持                                                   │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                ↓                                         │
│  ┌─────────────┬─────────────┬─────────────┬──────────────┬─────────┐  │
│  │   Scraper   │  Ingestor   │   Query     │  Debuginfo   │ Share   │  │
│  │   Manager   │             │   Service   │   Service    │ Service │  │
│  └─────────────┴─────────────┴─────────────┴──────────────┴─────────┘  │
│         ↓             ↓              ↓              ↓                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │              FrostDB Columnar Storage Engine                      │  │
│  │  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────┐  │  │
│  │  │ Active Memory   │  │     WAL      │  │  Parquet Files     │  │  │
│  │  │ (Arrow Tables)  │→ │  (Optional)  │→ │  (Object Storage)  │  │  │
│  │  │  512MB Default  │  │              │  │                    │  │  │
│  │  └─────────────────┘  └──────────────┘  └────────────────────┘  │  │
│  │                              ↓                                     │  │
│  │  Schema: stacktraces table with columns:                          │  │
│  │  - name, sample_type, sample_unit, period_type, period_unit       │  │
│  │  - stacktrace (LIST<STRING>), value, timestamp, duration          │  │
│  │  - labels.* (dynamic columns)                                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                ↓                                         │
│  ┌──────────────────────┬─────────────────────┬────────────────────┐   │
│  │   Symbolizer         │    BadgerDB KV      │   Object Storage   │   │
│  │   - DWARF            │    - Metastore      │   - Debuginfo      │   │
│  │   - pclntab          │    - Sym Cache      │   - Parquet Blocks │   │
│  │   - symtab           │    - Config         │   - Sources        │   │
│  │   - addr2line        │                     │                    │   │
│  └──────────────────────┴─────────────────────┴────────────────────┘   │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.2 运行模式

Parca 支持三种运行模式，通过 `--mode` 参数控制：

#### 4.2.1 All Mode（默认）

完整的 Parca 服务器，包含所有组件：

```
组件：
✓ Scraper Manager     # 抓取性能剖析数据
✓ ProfileStore        # 存储性能剖析数据
✓ Query Service       # 查询和可视化
✓ Symbolizer          # 符号化引擎
✓ Debuginfo Service   # 调试信息管理
✓ FrostDB             # 列式存储
✓ BadgerDB            # 元数据存储
✓ UI Server           # Web 界面

用途：单机部署或中心化服务器
```

#### 4.2.2 Scraper-Only Mode

仅运行抓取器，将数据转发到远程 Parca 服务器：

```
组件：
✓ Scraper Manager     # 抓取性能剖析数据
✓ gRPC Client         # 转发到远程服务器
✗ 存储/查询/符号化等组件

配置：
--mode=scraper-only
--store-address=remote-parca:7070
--bearer-token=xxx  # 可选认证

用途：边缘采集节点，减少本地资源消耗
```

#### 4.2.3 Forwarder Mode

接收数据并转发到远程服务器（中继模式）：

```
组件：
✓ ProfileStore        # 接收数据
✓ gRPC Client         # 转发数据
✗ Scraper/查询/符号化等组件

用途：多级部署架构，区域聚合
```

### 4.3 服务生命周期管理

Parca 使用 **`oklog/run`** 包管理服务组，实现优雅启动和关闭：

```go
// pkg/parca/parca.go 关键代码段
g := &run.Group{}

// 1. 添加信号处理器
g.Add(run.SignalHandler(ctx, syscall.SIGINT, syscall.SIGTERM))

// 2. 添加配置重载器
g.Add(func() error {
    <-configReloadChan
    // 重新加载配置
}, func(error) {
    close(configReloadChan)
})

// 3. 添加 HTTP 服务器
g.Add(func() error {
    return httpServer.ListenAndServe()
}, func(error) {
    httpServer.Shutdown(context.Background())
})

// 4. 添加符号化器
g.Add(func() error {
    symbolizer.Run(ctx)
}, func(error) {
    symbolizer.Stop()
})

// 启动所有服务（任何一个退出，全部退出）
return g.Run()
```

**优势**：
- 统一的错误处理
- 优雅关闭（graceful shutdown）
- 任意组件失败时，协调关闭所有组件

### 4.4 数据模型

#### 4.4.1 性能剖析数据 Schema

Parca 将 pprof 数据转换为 Apache Arrow 列式格式，Schema 定义在 `pkg/profile/schema.go`：

```go
// 核心列（所有性能剖析共有）
name         STRING  RLE_DICTIONARY      # 性能剖析名称，如 "process_cpu"
sample_type  STRING  RLE_DICTIONARY      # 样本类型，如 "cpu"
sample_unit  STRING  RLE_DICTIONARY      # 样本单位，如 "nanoseconds"
period_type  STRING  RLE_DICTIONARY      # 周期类型
period_unit  STRING  RLE_DICTIONARY      # 周期单位
period       INT64   RLE_DICTIONARY      # 周期值

stacktrace   LIST<STRING>  RLE_DICTIONARY  LZ4  NULLABLE  # 编码的调用栈
value        INT64         DELTA_BINARY_PACKED  LZ4       # 样本值
duration     INT64         RLE_DICTIONARY                 # 持续时间（delta profiles）
timestamp    INT64         DELTA_BINARY_PACKED  LZ4       # Unix 毫秒时间戳
time_nanos   INT64         DELTA_BINARY_PACKED  LZ4       # Unix 纳秒时间戳

// 动态标签列（pprof labels）
labels.*     STRING  RLE_DICTIONARY  # 例如 labels.job, labels.instance
```

**排序顺序**：
```
(name, sample_type, sample_unit, period_type, period_unit, timestamp, time_nanos)
```

这种排序确保：
1. 相同类型的性能剖析数据聚集在一起
2. 时间范围查询高效（时间戳排序）
3. 压缩效果最优（相同值连续出现）

#### 4.4.2 Stacktrace 编码格式

为最小化存储空间，调用栈采用紧凑的二进制编码：

```
每个 Location（栈帧）编码为：
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│  BuildID Len │  BuildID     │   Mapping    │   Address    │  Function    │
│  (varint)    │  (bytes)     │  (4 fields)  │  (uint64)    │  (optional)  │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

Mapping 编码：
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  File Len    │  File        │  Start Addr  │  Limit Addr  │
│  (varint)    │  (string)    │  (uint64)    │  (uint64)    │
└──────────────┴──────────────┴──────────────┴──────────────┘
│  Offset      │  HasFuncs    │  HasFilenames│  HasLineNum  │
│  (uint64)    │  (bool)      │  (bool)      │  (bool)      │
└──────────────┴──────────────┴──────────────┴──────────────┘

符号化后的 Location 额外包含：
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Lines Count │  Line 1      │  Line 2      │  ...         │
│  (varint)    │  (struct)    │  (struct)    │              │
└──────────────┴──────────────┴──────────────┴──────────────┘

Line 编码：
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Function    │  Line Number │  Column      │              │
│  Name Len    │  (int64)     │  (int64)     │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
│  Function    │  System Name │  Filename    │              │
│  Name        │  Len         │  Len         │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
│  System Name │  Filename    │  Start Line  │              │
│  (string)    │  (string)    │  (int64)     │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**优势**：
- **延迟符号化**：可存储未符号化的地址，查询时再符号化
- **去重**：RLE 字典编码对重复的调用栈非常有效
- **压缩**：LZ4 压缩在此编码基础上进一步减少空间

### 4.5 并发模型

#### 4.5.1 抓取并发

每个抓取目标运行在独立的 goroutine 中：

```
ScrapePool.sync() 每 5 秒执行一次
    ↓
创建/销毁 Target goroutines
    ↓
每个 Target 独立循环：
    for {
        sleep(scrapeInterval - lastScrapeDuration)
        scrape()  # HTTP 请求 + WriteRaw()
    }
```

**并发控制**：
- 每个 Target 独立，互不阻塞
- HTTP 超时控制（默认 10s）
- 错误不影响其他 Target

#### 4.5.2 查询并发

FrostDB 查询引擎内部并行化：

```
Query 请求
    ↓
构建逻辑计划（Filter, Aggregate, Project）
    ↓
执行引擎并行扫描：
    - 并行读取 Parquet 文件
    - 并行扫描内存表
    - 并行应用过滤器
    ↓
合并结果 → Arrow RecordBatch
```

**并发参数**：
- 由 FrostDB 内部管理，基于 `GOMAXPROCS`
- 查询间隔离（每个查询独立上下文）

#### 4.5.3 符号化并发

符号化器并发处理多个地址：

```go
// pkg/symbolizer/symbolizer.go
func (s *Symbolizer) Symbolize(ctx context.Context, stacktraces ...) {
    // 1. 提取所有唯一的 (BuildID, Address) 对
    uniqueAddrs := extractUniqueAddresses(stacktraces)

    // 2. 检查缓存（BadgerDB）
    cached, missing := s.checkCache(uniqueAddrs)

    // 3. 并发符号化缺失的地址
    results := make(chan SymbolizationResult)
    for _, addr := range missing {
        go func(a Address) {
            result := s.symbolizeOne(a)  // DWARF/pclntab/symtab
            results <- result
        }(addr)
    }

    // 4. 收集结果并缓存
    for range missing {
        result := <-results
        s.cache.Store(result)
    }
}
```

### 4.6 配置管理

#### 4.6.1 配置文件结构

`parca.yaml` 配置文件采用与 Prometheus 类似的结构：

```yaml
# 对象存储配置
object_storage:
  bucket:
    type: S3  # S3, GCS, AZURE, FILESYSTEM
    config:
      bucket: "parca-profiles"
      endpoint: "s3.amazonaws.com"
      access_key: "xxx"
      secret_key: "xxx"

# 抓取配置（类似 Prometheus）
scrape_configs:
  - job_name: "kubernetes-pods"
    scrape_interval: 10s
    scrape_timeout: 5s

    # Kubernetes 服务发现
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ["default", "production"]

    # 重新标记（relabeling）
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_parca_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod

    # 性能剖析配置
    profiling_config:
      pprof_config:
        # CPU 性能剖析（delta 类型）
        cpu:
          enabled: true
          path: /debug/pprof/profile
          delta: true

        # 内存性能剖析（绝对值类型）
        memory:
          enabled: true
          path: /debug/pprof/heap
          keep_sample_type:
            - type: inuse_space
              unit: bytes

  - job_name: "static-targets"
    static_configs:
      - targets:
          - "localhost:7070"
          - "app-server:8080"
        labels:
          environment: "production"
```

#### 4.6.2 配置热重载

配置文件变更时自动重载，无需重启服务：

```go
// pkg/parca/parca.go
configReloader := func() {
    watcher := fsnotify.NewWatcher()
    watcher.Add(configPath)

    for {
        select {
        case <-watcher.Events:
            newConfig := loadConfig(configPath)
            scrapeManager.ApplyConfig(newConfig)  // 热重载抓取配置

        case <-ctx.Done():
            return
        }
    }
}

g.Add(configReloader, ...)
```

**支持热重载的配置**：
- ✓ `scrape_configs`：抓取目标和间隔
- ✗ `object_storage`：需重启（涉及存储后端）
- ✗ `--storage-*` flags：需重启

---

## 5. 数据流与工作原理

### 5.1 完整数据流图

```
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 1: Collection (数据采集)                                       │
└─────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────┐
│  Target Process    │  (例如: Go HTTP Server with /debug/pprof)
│  Port 8080         │
└────────────────────┘
        ↑ HTTP GET /debug/pprof/profile?seconds=10
        │
┌────────────────────┐
│  Scraper Loop      │  (每 10s 执行一次)
│  (goroutine)       │
└────────────────────┘
        ↓ pprof bytes (protobuf)
        │
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 2: Normalization (数据规范化)                                  │
└─────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  normalizer.WriteRawRequestToArrowRecord()                          │
│  1. 解析 pprof.Profile (google/pprof 库)                           │
│  2. 提取关键字段:                                                   │
│     - SampleType: [{Type: "cpu", Unit: "nanoseconds"}]            │
│     - PeriodType: {Type: "cpu", Unit: "nanoseconds"}              │
│     - Period: 10000000 (10ms)                                      │
│     - Samples: []Sample{                                           │
│         {Location: [0x4a2f30, 0x4a2e10, ...], Value: [100000]}    │
│       }                                                             │
│  3. 编码 Stacktrace:                                               │
│     每个 Location → [BuildID][Mapping][Address]                    │
│  4. 构建 Arrow RecordBatch:                                        │
│     Columns: [name, sample_type, ..., stacktrace, value, ...]    │
└────────────────────────────────────────────────────────────────────┘
        ↓ Arrow RecordBatch
        │
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 3: Storage (数据存储)                                          │
└─────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  FrostDB Table.InsertRecord()                                       │
│  1. Schema 验证                                                     │
│  2. 排序: (name, sample_type, ..., timestamp)                      │
│  3. 插入 Active Memory Buffer (Arrow Table)                         │
│  4. 如果启用 WAL: 写入 Write-Ahead Log                              │
│  5. 触发压缩条件检查:                                               │
│     - Active Memory > 512MB                                        │
│     - WAL Size > Snapshot Trigger (128MB)                          │
│     - 定时压缩 (每 5 分钟)                                          │
└────────────────────────────────────────────────────────────────────┘
        ↓ (当触发压缩时)
        │
┌────────────────────────────────────────────────────────────────────┐
│  Compaction Process (压缩过程)                                      │
│  1. 创建 Parquet Writer                                            │
│  2. 编码选项:                                                       │
│     - RLE Dictionary: name, sample_type, labels.*                 │
│     - Delta Encoding: timestamp, value                            │
│     - LZ4 Compression: stacktrace, value                          │
│  3. 写入 Row Groups (默认 8192 行/组)                              │
│  4. 上传到 Object Storage                                          │
│  5. 更新 FrostDB 索引 (Block ID → File Path)                      │
│  6. 释放 Active Memory                                             │
└────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  Object Storage (S3/GCS/...)                                        │
│  /parca-profiles/                                                  │
│    ├── blocks/                                                     │
│    │   ├── 01J1A2B3C4D5E6F7/                                      │
│    │   │   └── stacktraces.parquet  (压缩后: 50-100x)             │
│    │   └── 01J1A2B3C4D5E6F8/                                      │
│    ├── debuginfo/                                                  │
│    └── sources/                                                    │
└────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Phase 4: Query (数据查询)                                            │
└─────────────────────────────────────────────────────────────────────┘
        ↑ gRPC Query Request
        │
┌────────────────────────────────────────────────────────────────────┐
│  QueryService.Query()                                               │
│  Request: {                                                         │
│    Query: "process_cpu:cpu:nanoseconds:cpu:nanoseconds{job='app'}"│
│    ReportType: FLAMEGRAPH_ARROW                                    │
│    TimeRange: [start, end]                                         │
│  }                                                                  │
└────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  parcacol.Querier.QueryRange()                                     │
│  1. 解析查询字符串 → QueryParts:                                   │
│     Meta: {name: "process_cpu", sample_type: "cpu", ...}          │
│     Matchers: [job="app"]                                          │
│  2. 构建 FrostDB Logical Plan:                                     │
│     ScanTable("stacktraces")                                       │
│       .Filter(name == "process_cpu" AND labels.job == "app"       │
│              AND time_nanos >= start AND time_nanos <= end)        │
│       .Aggregate(SUM(value * period), GROUP BY stacktrace)         │
│       .Execute()                                                   │
└────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  FrostDB Query Execution                                            │
│  1. 扫描 Parquet Files (并行读取多个文件)                          │
│  2. 应用 Filter (谓词下推到 Parquet Reader)                        │
│  3. 扫描 Active Memory Buffer                                       │
│  4. Merge Results → Arrow RecordBatch:                             │
│     [stacktrace] → [aggregated_value]                              │
│     [[addr1, addr2, addr3]] → [1234567890]  (纳秒)                │
└────────────────────────────────────────────────────────────────────┘
        ↓ Arrow RecordBatch (未符号化)
        │
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 5: Symbolization (符号化)                                      │
└─────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  Symbolizer.Symbolize()                                            │
│  1. 提取唯一的 (BuildID, Address) 对:                             │
│     - BuildID: "abc123..."                                         │
│     - Addresses: [0x4a2f30, 0x4a2e10, ...]                        │
│  2. 检查符号化缓存 (BadgerDB):                                     │
│     Key: BuildID + Address                                         │
│     Value: {FunctionName, Filename, LineNumber}                    │
│  3. 缺失的地址:                                                     │
│     a) 查询 Debuginfo Service → 获取调试信息文件                   │
│     b) 如果本地无: 查询 debuginfod.elfutils.org                    │
│     c) 下载并缓存到 Object Storage                                 │
│  4. 创建 Liner (基于调试信息类型):                                 │
│     - DWARF: DWARFLiner (使用 go-delve/delve)                     │
│     - Go pclntab: PclntabLiner                                     │
│     - ELF symtab: SymtabLiner                                      │
│  5. 解析地址:                                                       │
│     Addr 0x4a2f30 → {                                              │
│       Function: "main.processRequest",                             │
│       File: "/app/main.go",                                        │
│       Line: 42                                                     │
│     }                                                               │
│  6. 缓存结果到 BadgerDB                                             │
└────────────────────────────────────────────────────────────────────┘
        ↓ 符号化后的 Arrow RecordBatch
        │
┌─────────────────────────────────────────────────────────────────────┐
│ Phase 6: Report Generation (报告生成)                                │
└─────────────────────────────────────────────────────────────────────┘
        ↓
┌────────────────────────────────────────────────────────────────────┐
│  GenerateFlamegraphArrow()                                          │
│  1. 构建树结构 (从叶子到根):                                       │
│     Root                                                            │
│      ├─ main.main (1000ms, 100%)                                   │
│      │   ├─ main.processRequest (800ms, 80%)                       │
│      │   │   ├─ json.Unmarshal (400ms, 40%)                        │
│      │   │   └─ db.Query (300ms, 30%)                              │
│      │   └─ main.logRequest (100ms, 10%)                           │
│      └─ runtime.goexit (100ms, 10%)                                │
│  2. 计算累积值 (cumulative) 和差异值 (diff, 如果对比模式)         │
│  3. 修剪小于阈值的节点 (默认 0.1%)                                 │
│  4. 编码为 Arrow RecordBatch:                                      │
│     Columns: [cumulative, function_name, file_name, line_number]  │
└────────────────────────────────────────────────────────────────────┘
        ↓ Flamegraph Arrow Response
        │
┌────────────────────────────────────────────────────────────────────┐
│  gRPC Response → HTTP/JSON (via grpc-gateway)                       │
│  {                                                                  │
│    "flamegraph": {                                                 │
│      "root": { "cumulative": 1000, "children": [...] }            │
│    },                                                               │
│    "total": 1000,                                                  │
│    "filtered": 1000                                                │
│  }                                                                  │
└────────────────────────────────────────────────────────────────────┘
```

### 5.2 关键数据转换步骤

#### 5.2.1 pprof → Arrow 转换

**输入** (pprof.Profile):
```protobuf
message Profile {
  repeated ValueType sample_type = 1;
  repeated Sample sample = 2;
  repeated Mapping mapping = 3;
  repeated Location location = 4;
  repeated Function function = 5;
  repeated string string_table = 6;
  int64 time_nanos = 9;
  int64 duration_nanos = 10;
  ValueType period_type = 11;
  int64 period = 12;
}
```

**输出** (Arrow RecordBatch):
```
Arrow Schema:
  name: string (dictionary)
  sample_type: string (dictionary)
  sample_unit: string (dictionary)
  period_type: string (dictionary)
  period_unit: string (dictionary)
  period: int64
  stacktrace: list<string>  (每个 string 是编码的 Location)
  value: int64
  duration: int64
  timestamp: int64
  time_nanos: int64
  labels.<key>: string (动态列)

示例数据:
Row 0:
  name = "process_cpu"
  sample_type = "cpu"
  sample_unit = "nanoseconds"
  period = 10000000
  stacktrace = [[BuildID1][Mapping1][0x4a2f30], [BuildID1][Mapping1][0x4a2e10], ...]
  value = 10000000  (10ms CPU 时间)
  time_nanos = 1735804800000000000
  labels.job = "my-app"
  labels.instance = "pod-123"
```

#### 5.2.2 Delta 与非 Delta 性能剖析处理

**Delta Profiles** (CPU, goroutine):
- 样本值表示**增量**（自上次抓取以来的变化）
- 需要 `duration` 字段表示采集时长
- 查询时计算**每秒速率**：

```go
valuePerSecond = sum(value * period) / sum(duration)

// 示例:
// Sample 1: value=100, period=10ms, duration=10s
// Sample 2: value=200, period=10ms, duration=10s
// Total: (100*10ms + 200*10ms) / (10s + 10s) = 3ms / 20s = 0.15ms/s
```

**Non-Delta Profiles** (Heap, allocs):
- 样本值表示**绝对值**（当前状态快照）
- 不需要 `duration` 字段
- 查询时直接聚合：

```go
totalValue = sum(value * period)

// 示例: Heap inuse_space
// Sample 1: value=1000, period=1 → 1000 bytes
// Sample 2: value=2000, period=1 → 2000 bytes
// Total: 3000 bytes
```

### 5.3 符号化工作流详解

```
┌──────────────────────────────────────────────────────────────┐
│  Step 1: 提取 Build ID 和地址                                 │
└──────────────────────────────────────────────────────────────┘
输入: Stacktrace [[BuildID1][Mapping][0x4a2f30], [BuildID1][Mapping][0x4a2e10]]
      ↓
提取: {
  BuildID: "abc123def456...",
  Addresses: [0x4a2f30, 0x4a2e10]
}

┌──────────────────────────────────────────────────────────────┐
│  Step 2: 地址规范化 (PIE 可执行文件)                          │
└──────────────────────────────────────────────────────────────┘
if ELF_Type == ET_DYN:  // Position Independent Executable
    normalized_addr = runtime_addr - mapping_start + mapping_offset
else:  // Fixed address executable
    normalized_addr = runtime_addr

示例:
  runtime_addr = 0x5643a2f30
  mapping_start = 0x564300000
  mapping_offset = 0x100000
  → normalized = 0x5643a2f30 - 0x564300000 + 0x100000 = 0x10a2f30

┌──────────────────────────────────────────────────────────────┐
│  Step 3: 查询符号化缓存 (BadgerDB)                            │
└──────────────────────────────────────────────────────────────┘
Key: BuildID + ":" + hex(normalized_addr)
     "abc123...:0x10a2f30"
      ↓
Value (如果缓存命中): {
  FunctionName: "main.processRequest",
  Filename: "/app/main.go",
  LineNumber: 42
}
Cache Hit? → 直接返回
Cache Miss? → 继续 Step 4

┌──────────────────────────────────────────────────────────────┐
│  Step 4: 获取调试信息                                          │
└──────────────────────────────────────────────────────────────┘
1. 查询本地 Object Storage:
   /debuginfo/<BuildID>/executable
   /debuginfo/<BuildID>/debuginfo  (DWARF separate file)

2. 如果本地无, 查询 Debuginfod:
   GET http://debuginfod.elfutils.org/buildid/<BuildID>/debuginfo
   Response: Binary DWARF data
   → 缓存到 Object Storage

3. 如果 Debuginfod 无:
   → 等待用户上传 (通过 DebugInfo Service)

┌──────────────────────────────────────────────────────────────┐
│  Step 5: 创建 Liner (地址解析器)                               │
└──────────────────────────────────────────────────────────────┘
检查调试信息类型 (优先级顺序):

1. DWARF (.debug_info, .debug_line 段):
   → DWARFLiner (最精确, 支持内联函数)

2. Go pclntab (.gopclntab 段):
   → PclntabLiner (Go 特有, 快速)

3. ELF symtab/dynsym:
   → SymtabLiner (仅函数名, 无行号)

4. 外部 addr2line:
   → ExternalAddr2lineLiner (备选方案)

┌──────────────────────────────────────────────────────────────┐
│  Step 6: 解析地址                                              │
└──────────────────────────────────────────────────────────────┘
liner.PCToLines(normalized_addr) → []Line{
  {
    Function: {
      Name: "main.processRequest",
      SystemName: "main.processRequest",  (C++ demangled)
      Filename: "/build/app/main.go",
      StartLine: 40
    },
    Line: 42,
    Column: 0
  }
}

如果是内联函数, 返回多行:
  [
    {Function: "inner", Line: 10},   // 内联函数
    {Function: "outer", Line: 42}    // 调用点
  ]

┌──────────────────────────────────────────────────────────────┐
│  Step 7: C++ Symbol Demangling                                │
└──────────────────────────────────────────────────────────────┘
输入: "_ZN3fooC1Ev"
      ↓
ianlancetaylor/demangle.ToString()
      ↓
输出 (根据模式):
  - simple: "foo"  (无参数, 无模板, 无返回类型)
  - templates: "foo<T>"
  - full: "foo::foo()"

┌──────────────────────────────────────────────────────────────┐
│  Step 8: 缓存结果                                              │
└──────────────────────────────────────────────────────────────┘
badgerDB.Set(
  Key: "abc123...:0x10a2f30",
  Value: encode({
    FunctionName: "main.processRequest",
    Filename: "/app/main.go",
    LineNumber: 42
  })
)
```

### 5.4 查询优化策略

#### 5.4.1 谓词下推 (Predicate Pushdown)

FrostDB 将过滤条件下推到 Parquet 读取层：

```
查询: WHERE labels.job = 'my-app' AND timestamp >= 1000 AND timestamp <= 2000

传统执行:
  1. 读取所有列的所有行
  2. 在内存中应用过滤器
  → 慢, 内存消耗大

优化后:
  1. Parquet Reader 读取 Row Group metadata
  2. 跳过不匹配的 Row Groups (基于 Min/Max statistics)
  3. 仅读取匹配的列 (labels.job, timestamp, stacktrace, value)
  4. 应用剩余过滤器
  → 快, 内存消耗小
```

#### 5.4.2 列剪枝 (Column Pruning)

仅读取查询需要的列：

```
查询: SELECT stacktrace, value WHERE labels.job = 'my-app'

传统执行:
  读取所有列: [name, sample_type, ..., stacktrace, value, ...]
  → 浪费 I/O 和内存

优化后:
  仅读取: [labels.job, stacktrace, value]
  → 减少 80% 的 I/O
```

#### 5.4.3 懒符号化 (Lazy Symbolization)

仅在需要显示时符号化：

```
Flamegraph 查询 (需要符号化):
  Query → FrostDB → 未符号化 RecordBatch → Symbolizer → 符号化 RecordBatch → Flamegraph

QueryRange 时间序列 (不需要符号化):
  Query → FrostDB → 未符号化 RecordBatch → Aggregate by labels → 返回
  → 跳过符号化, 节省 90% 的时间
```

---

## 6. 核心模块详解

### 6.1 Scraper Manager (pkg/scrape/)

**责任**：管理所有抓取任务的生命周期

#### 6.1.1 Manager 结构

```go
// pkg/scrape/manager.go
type Manager struct {
    opts     *Options
    logger   log.Logger
    scrapeConfigs chan *config.Config  // 配置变更通道

    targetSets map[string][]*targetgroup.Group  // 服务发现结果
    scrapePoolsMtx sync.Mutex
    scrapePools    map[string]*scrapePool  // Job Name → Scrape Pool

    triggerReload chan struct{}
}
```

#### 6.1.2 核心方法

**ApplyConfig**: 应用新配置，创建/更新/删除 Scrape Pool

```go
func (m *Manager) ApplyConfig(cfg *config.Config) error {
    // 1. 对比新旧配置, 找出需要创建/更新/删除的 Job
    toCreate, toUpdate, toDelete := diffConfigs(oldCfg, cfg)

    // 2. 删除不再存在的 Job
    for _, jobName := range toDelete {
        m.scrapePools[jobName].stop()
        delete(m.scrapePools, jobName)
    }

    // 3. 创建新 Job 的 Scrape Pool
    for _, jobName := range toCreate {
        sp := newScrapePool(cfg.ScrapeConfigs[jobName], m.profileStore)
        m.scrapePools[jobName] = sp
        sp.start()
    }

    // 4. 更新现有 Job 的配置
    for _, jobName := range toUpdate {
        m.scrapePools[jobName].reload(cfg.ScrapeConfigs[jobName])
    }
}
```

**Run**: 主循环，监听配置变更和服务发现事件

```go
func (m *Manager) Run(ctx context.Context) error {
    // 启动服务发现 (Prometheus Discovery Manager)
    discoveryMgr := discovery.NewManager(ctx, m.logger)
    go discoveryMgr.Run()

    for {
        select {
        case <-m.triggerReload:
            // 重新加载配置
            m.ApplyConfig(newConfig)

        case targetGroups := <-discoveryMgr.SyncCh():
            // 服务发现更新
            for jobName, groups := range targetGroups {
                m.scrapePools[jobName].Sync(groups)
            }

        case <-ctx.Done():
            return nil
        }
    }
}
```

#### 6.1.3 Scrape Pool

每个 Job 对应一个 Scrape Pool，管理该 Job 的所有 Target：

```go
type scrapePool struct {
    config *config.ScrapeConfig
    client *http.Client
    profileStore ProfileStore

    mtx     sync.Mutex
    targets map[uint64]*Target  // Target Hash → Target
    loops   map[uint64]loop     // Target Hash → Scrape Loop
}

func (sp *scrapePool) Sync(groups []*targetgroup.Group) {
    // 1. 应用 relabeling, 生成最终 Target 列表
    targets := relabel.Process(groups, sp.config.RelabelConfigs)

    // 2. 对比现有 Target, 找出新增/删除/更新的
    toAdd, toRemove := sp.diff(targets)

    // 3. 停止并删除已移除的 Target
    for hash, target := range toRemove {
        sp.loops[hash].stop()
        delete(sp.targets, hash)
        delete(sp.loops, hash)
    }

    // 4. 启动新 Target 的抓取循环
    for hash, target := range toAdd {
        l := newScrapeLoop(target, sp.client, sp.profileStore)
        sp.targets[hash] = target
        sp.loops[hash] = l
        go l.run()
    }
}
```

#### 6.1.4 Scrape Loop

每个 Target 独立的抓取循环：

```go
type scrapeLoop struct {
    target       *Target
    client       *http.Client
    profileStore ProfileStore
    interval     time.Duration
    timeout      time.Duration

    stopCh chan struct{}
}

func (sl *scrapeLoop) run() {
    ticker := time.NewTicker(sl.interval)
    defer ticker.Stop()

    for {
        select {
        case <-ticker.C:
            start := time.Now()
            err := sl.scrape()
            scrapeDuration := time.Since(start)

            // 记录指标
            targetScrapeDuration.WithLabelValues(sl.target.Labels()...).Observe(scrapeDuration.Seconds())
            if err != nil {
                targetScrapeErrors.WithLabelValues(sl.target.Labels()...).Inc()
            }

        case <-sl.stopCh:
            return
        }
    }
}

func (sl *scrapeLoop) scrape() error {
    // 1. 构建 HTTP 请求
    req, _ := http.NewRequest("GET", sl.target.URL(), nil)
    req.Header.Set("User-Agent", "Parca/1.0")

    // 2. 执行请求 (带超时)
    ctx, cancel := context.WithTimeout(context.Background(), sl.timeout)
    defer cancel()
    resp, err := sl.client.Do(req.WithContext(ctx))
    if err != nil {
        return err
    }
    defer resp.Body.Close()

    // 3. 读取 pprof 数据
    pprofData, _ := io.ReadAll(resp.Body)

    // 4. 写入 ProfileStore
    return sl.profileStore.WriteRaw(ctx, &profilestorepb.WriteRawRequest{
        Series: []*profilestorepb.RawProfileSeries{{
            Labels: &profilestorepb.LabelSet{
                Labels: sl.target.Labels(),
            },
            Samples: []*profilestorepb.RawSample{{
                RawProfile: pprofData,
            }},
        }},
    })
}
```

### 6.2 ProfileStore (pkg/profilestore/)

**责任**：接收性能剖析数据，规范化并存储到 FrostDB

#### 6.2.1 核心接口

```go
type ProfileColumnStore struct {
    logger     log.Logger
    tracer     trace.Tracer
    ingester   Ingester  // FrostDB 表插入器
    normalizer Normalizer // pprof → Arrow 转换器
    table      frostdb.Table

    // 双向流 (Agent 模式)
    mtx            sync.RWMutex
    agents         map[string]*agentWriteClient
}
```

#### 6.2.2 WriteRaw 实现

```go
func (s *ProfileColumnStore) WriteRaw(ctx context.Context, req *pb.WriteRawRequest) (*pb.WriteRawResponse, error) {
    // 1. 转换 pprof 数据为 Arrow RecordBatch
    records, err := s.normalizer.WriteRawRequestToArrowRecord(ctx, req)
    if err != nil {
        return nil, err
    }

    // 2. 批量插入 FrostDB
    for _, record := range records {
        if err := s.ingester.Ingest(ctx, record); err != nil {
            return nil, err
        }
    }

    return &pb.WriteRawResponse{}, nil
}
```

#### 6.2.3 Normalizer (pkg/normalizer/)

**WriteRawRequestToArrowRecord** 的关键步骤：

```go
func (n *Normalizer) WriteRawRequestToArrowRecord(ctx context.Context, req *pb.WriteRawRequest) ([]arrow.Record, error) {
    records := []arrow.Record{}

    for _, series := range req.Series {
        labels := parseLabelSet(series.Labels)

        for _, sample := range series.Samples {
            // 1. 解析 pprof.Profile
            prof, err := profile.ParseData(sample.RawProfile)
            if err != nil {
                continue
            }

            // 2. 提取 Sample Type (可能有多个, 如 alloc_space, alloc_objects)
            for _, sampleType := range prof.SampleType {
                // 3. 构建 Arrow Builder
                builder := array.NewRecordBuilder(memory.DefaultAllocator, n.schema)

                // 4. 添加元数据列
                builder.Field(0).(*array.StringBuilder).Append(prof.Name)  // name
                builder.Field(1).(*array.StringBuilder).Append(sampleType.Type)  // sample_type
                builder.Field(2).(*array.StringBuilder).Append(sampleType.Unit)  // sample_unit

                // 5. 处理每个 Sample
                for _, sample := range prof.Sample {
                    // 编码 Stacktrace
                    encodedStacktrace := n.encodeStacktrace(sample.Location)
                    builder.Field(6).(*array.ListBuilder).Append(true)
                    for _, loc := range encodedStacktrace {
                        builder.Field(6).(*array.ListBuilder).ValueBuilder().(*array.StringBuilder).Append(loc)
                    }

                    // 样本值
                    value := sample.Value[sampleTypeIndex]
                    builder.Field(7).(*array.Int64Builder).Append(value)

                    // 时间戳
                    builder.Field(8).(*array.Int64Builder).Append(prof.TimeNanos / 1e6)  // timestamp (ms)
                    builder.Field(9).(*array.Int64Builder).Append(prof.TimeNanos)       // time_nanos

                    // 动态标签
                    for key, value := range labels {
                        fieldIndex := n.schema.FieldIndices("labels." + key)[0]
                        builder.Field(fieldIndex).(*array.StringBuilder).Append(value)
                    }
                }

                // 6. 生成 RecordBatch
                record := builder.NewRecord()
                records = append(records, record)
            }
        }
    }

    return records, nil
}
```

#### 6.2.4 Stacktrace 编码实现

```go
func (n *Normalizer) encodeStacktrace(locations []*profile.Location) []string {
    encoded := make([]string, len(locations))

    for i, loc := range locations {
        var buf bytes.Buffer

        // BuildID (varint length + bytes)
        buildID := loc.Mapping.BuildID
        binary.Write(&buf, binary.LittleEndian, uint64(len(buildID)))
        buf.WriteString(buildID)

        // Mapping
        binary.Write(&buf, binary.LittleEndian, uint64(len(loc.Mapping.File)))
        buf.WriteString(loc.Mapping.File)
        binary.Write(&buf, binary.LittleEndian, loc.Mapping.Start)
        binary.Write(&buf, binary.LittleEndian, loc.Mapping.Limit)
        binary.Write(&buf, binary.LittleEndian, loc.Mapping.Offset)

        // Address
        binary.Write(&buf, binary.LittleEndian, loc.Address)

        // Function Name (如果已符号化)
        if len(loc.Line) > 0 {
            funcName := loc.Line[0].Function.Name
            binary.Write(&buf, binary.LittleEndian, uint64(len(funcName)))
            buf.WriteString(funcName)
        }

        encoded[i] = buf.String()
    }

    return encoded
}
```

---

## 7. 总结与关键技术点

### 7.1 核心技术亮点

1. **列式存储优化**
   - Apache Arrow 零拷贝内存格式
   - Parquet 高压缩比持久化（50-100x）
   - 字典编码 + Delta 编码 + LZ4 压缩

2. **延迟符号化架构**
   - 存储未符号化地址，节省存储空间
   - 查询时按需符号化，减少不必要的开销
   - BadgerDB 缓存符号化结果，加速重复查询

3. **高效查询引擎**
   - 谓词下推（Predicate Pushdown）
   - 列剪枝（Column Pruning）
   - 并行扫描 Parquet 文件

4. **灵活的部署模式**
   - All Mode：完整服务器
   - Scraper-Only Mode：边缘采集
   - Forwarder Mode：多级架构

5. **云原生设计**
   - 无状态服务（可水平扩展）
   - 对象存储后端（S3/GCS/Azure）
   - Kubernetes 原生集成

### 7.2 性能特征总结

| 指标 | 数值 | 说明 |
|-----|------|------|
| 写入吞吐量 | ~10K profiles/s | 单实例 |
| 写入延迟 P99 | < 100ms | WriteRaw 请求 |
| 冷查询延迟 | 1-5s | 包含符号化 |
| 热查询延迟 | 100-500ms | 符号化缓存命中 |
| 存储压缩比 | 50-100x | 相比原始 pprof |
| 内存占用 | 512MB-2GB | 默认配置 |
| CPU 开销 | < 1% | 抓取器对目标的影响 |

### 7.3 构建与部署快速参考

**本地开发**：
```bash
# 构建（UI + Go 二进制）
make build

# 运行
./bin/parca

# 访问 UI
open http://localhost:7070
```

**Docker 部署**：
```bash
docker run -p 7070:7070 \
  -v $(pwd)/parca.yaml:/etc/parca/parca.yaml \
  ghcr.io/parca-dev/parca:latest
```

**Kubernetes 部署**：
```bash
kubectl apply -f deploy/kubernetes/
```

### 7.4 关键文件索引

| 文件路径 | 行数 | 说明 |
|---------|------|------|
| `cmd/parca/main.go` | 72 | 程序入口 |
| `pkg/parca/parca.go` | ~900 | 主服务协调器 |
| `pkg/scrape/manager.go` | ~300 | 抓取管理器 |
| `pkg/profilestore/profilecolumnstore.go` | ~250 | 存储服务 |
| `pkg/query/columnquery.go` | ~1200 | 查询服务 |
| `pkg/parcacol/querier.go` | ~1200 | FrostDB 查询器 |
| `pkg/symbolizer/symbolizer.go` | ~800 | 符号化引擎 |
| `pkg/profile/schema.go` | ~200 | Arrow Schema 定义 |
| `proto/parca/query/v1alpha1/query.proto` | ~200 | 查询 API 定义 |

### 7.5 依赖关系图

```
Parca Core
├── FrostDB (列式存储引擎)
│   ├── Apache Arrow (内存格式)
│   ├── Parquet-go (持久化)
│   └── DynParquet (动态 Schema)
├── BadgerDB (KV 存储)
│   ├── 符号化缓存
│   └── Debuginfo 元数据
├── Prometheus 生态
│   ├── prometheus/prometheus (服务发现)
│   └── prometheus/client_golang (指标)
├── gRPC + grpc-gateway
│   ├── google.golang.org/grpc
│   └── grpc-ecosystem/grpc-gateway
├── 符号化工具链
│   ├── go-delve/delve (DWARF)
│   ├── google/pprof (pprof 解析)
│   └── ianlancetaylor/demangle (C++ demangling)
└── 对象存储
    └── thanos-io/objstore (统一抽象层)
```

### 7.6 扩展与定制建议

**添加新的性能剖析类型**：
1. 修改 `pkg/profile/schema.go`：添加新的 sample_type
2. 修改 `pkg/normalizer/normalizer.go`：处理新格式
3. 更新 UI：`ui/packages/shared/profile/`

**添加新的存储后端**：
1. 实现 `thanos-io/objstore.Bucket` 接口
2. 注册到 `pkg/parca/parca.go` 的对象存储初始化逻辑

**添加新的符号化后端**：
1. 实现 `pkg/symbolizer.Liner` 接口
2. 注册到 `pkg/symbolizer/symbolizer.go` 的 Liner 创建逻辑

**添加新的查询报告类型**：
1. 定义 `proto/parca/query/v1alpha1/query.proto` 中的新 ReportType
2. 实现 `pkg/query/` 中的报告生成器
3. 更新 UI 渲染逻辑

---

## 附录

### A. 常用命令速查

```bash
# 构建
make build              # 完整构建（UI + Go）
make go/build           # 仅构建 Go
make ui/build           # 仅构建 UI

# 测试
make test               # 运行所有测试
make go/test            # 仅 Go 测试
make ui/test            # 仅 UI 测试

# 代码质量
make lint               # 运行 lint
make format             # 格式化代码

# Protocol Buffers
make proto/generate     # 生成 protobuf 代码
make proto/lint         # lint proto 文件

# 容器化
make container          # 构建容器镜像
make push-container     # 推送镜像

# 开发环境
make dev/up             # 启动本地开发环境（Tilt）
make dev/down           # 停止本地开发环境
```

### B. 配置示例

**完整的 parca.yaml 配置**：
```yaml
object_storage:
  bucket:
    type: S3
    config:
      bucket: "parca-profiles"
      endpoint: "s3.amazonaws.com"
      region: "us-east-1"
      access_key: "${AWS_ACCESS_KEY_ID}"
      secret_key: "${AWS_SECRET_ACCESS_KEY}"

scrape_configs:
  # Kubernetes Pods
  - job_name: "kubernetes-pods"
    scrape_interval: 10s
    scrape_timeout: 5s
    kubernetes_sd_configs:
      - role: pod
        namespaces:
          names: ["default", "production"]
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_parca_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_name]
        target_label: pod
      - source_labels: [__meta_kubernetes_namespace]
        target_label: namespace
    profiling_config:
      pprof_config:
        cpu:
          enabled: true
          path: /debug/pprof/profile
          delta: true
        memory:
          enabled: true
          path: /debug/pprof/heap
          keep_sample_type:
            - type: inuse_space
              unit: bytes

  # Static Targets
  - job_name: "static-apps"
    scrape_interval: 30s
    static_configs:
      - targets:
          - "app1.example.com:8080"
          - "app2.example.com:8080"
        labels:
          environment: "production"
          team: "backend"
```

### C. 故障排查指南

**问题 1：符号化失败**
- 检查 debuginfo 是否已上传：`curl http://localhost:7070/api/v1/debuginfo/<build-id>`
- 检查 debuginfod 配置：`--debuginfod-upstream-servers`
- 查看符号化日志：`--log-level=debug`

**问题 2：查询慢**
- 检查时间范围是否过大：减小时间窗口
- 检查标签基数：避免高基数标签（如 request_id）
- 启用索引持久化：`--storage-index-on-disk`

**问题 3：内存占用高**
- 调整活跃内存大小：`--storage-active-memory=268435456` (256MB)
- 启用 WAL 和快照：`--storage-enable-wal`
- 启用磁盘索引：`--storage-index-on-disk`

**问题 4：抓取失败**
- 检查目标可达性：`curl http://target:port/debug/pprof/profile`
- 检查防火墙规则
- 查看抓取日志：访问 `/targets` 页面

### D. 相关资源

- **官方网站**: https://www.parca.dev/
- **GitHub**: https://github.com/parca-dev/parca
- **文档**: https://www.parca.dev/docs/
- **Discord 社区**: https://discord.gg/ZgUpYgpzXy
- **FrostDB**: https://github.com/polarsignals/frostdb
- **Apache Arrow**: https://arrow.apache.org/
- **pprof 格式规范**: https://github.com/google/pprof/blob/master/proto/profile.proto

---

**文档版本**: v1.0
**最后更新**: 2025-11-02
**基于 Parca 版本**: main branch (commit 25345811e)

