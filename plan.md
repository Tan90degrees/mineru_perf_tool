prompt: 我想要写一个mineru的吞吐量评测工具，输入可以是指定的目录，目录中可以是图片或者pdf文件，还有一个特殊模式是omnidocbench，支持输出评测精度。可以跑在NPU环境下，支持pipeline、vlm、hybrid模式，其中pipeline模式支持CPU模式，包括测试的客户端和服务端，服务端支持多卡多实例（可以有多张卡，每张卡可以有多个实例）支持指定NPU卡。支持网格搜索调参，可调整参数有：卡数量、每张卡的上的实例个数、请求并发数、请求组批大小

## 产品概述

MinerU吞吐量评测工具是一个用于测试和优化MinerU文档解析服务性能的综合工具。支持在NPU/GPU/CPU环境下运行，提供完整的服务端部署、客户端压测和精度评测功能。

## 核心功能

- **输入模式**：支持指定目录（PDF/图片文件）和OmniDocBench评测集模式
- **服务端部署**：基于现有FastAPI服务，支持多卡多实例部署，通过命令行参数指定NPU卡
- **客户端压测**：异步并发请求测试，支持组批处理
- **网格搜索调参**：自动测试不同参数组合（卡数量、实例个数、并发数、组批大小）
- **评测精度**：OmniDocBench模式下输出端到端评测精度
- **资源监控**：实时监控GPU/NPU显存占用、CPU/内存占用
- **结果输出**：终端实时输出 + JSON/CSV报告 + 可视化图表（热力图）

## 技术栈选择

- **语言**：Python 3.10+
- **异步框架**：asyncio + aiohttp（客户端并发请求）
- **服务框架**：FastAPI（复用MinerU现有服务）
- **进程管理**：multiprocessing + subprocess（多实例管理）
- **资源监控**：psutil（CPU/内存）+ torch/torch_npu（GPU/NPU显存）
- **数据处理**：pandas（结果分析）+ matplotlib/seaborn（可视化）
- **配置管理**：YAML + argparse

## 实现方案

### 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    mineru_perf_tool                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐  │
│  │   CLI入口   │───▶│  配置管理   │───▶│     任务调度器      │  │
│  └─────────────┘    └─────────────┘    └─────────────────────┘  │
│                                                 │                │
│         ┌───────────────────────────────────────┼──────────────┐ │
│         │                                       ▼              │ │
│         │  ┌─────────────┐    ┌─────────────────────────────┐  │ │
│         │  │ 服务端管理器 │───▶│  多卡多实例进程组            │  │ │
│         │  └─────────────┘    │  [NPU:0] [NPU:1] ...        │  │ │
│         │                     │  实例1  实例1                │  │ │
│         │                     │  实例2  实例2                │  │ │
│         │                     └─────────────────────────────┘  │ │
│         │                              │                        │ │
│         │                              ▼                        │ │
│         │  ┌─────────────┐    ┌─────────────────────────────┐  │ │
│         │  │ 客户端压测器 │◀───│     FastAPI服务实例组       │  │ │
│         │  └─────────────┘    │  http://host:port1           │  │ │
│         │         │           │  http://host:port2           │  │ │
│         │         ▼           └─────────────────────────────┘  │ │
│         │  ┌─────────────┐                                     │ │
│         │  │ 资源监控器  │                                     │ │
│         │  └─────────────┘                                     │ │
│         │         │                                            │ │
│         └─────────┼────────────────────────────────────────────┘ │
│                   ▼                                              │
│         ┌─────────────────────────────────────────────────────┐  │
│         │              结果分析与可视化                        │  │
│         │  ┌─────────┐ ┌─────────┐ ┌───────────────────────┐  │  │
│         │  │实时输出 │ │报告生成│ │  可视化图表(热力图)   │  │  │
│         │  └─────────┘ └─────────┘ └───────────────────────┘  │  │
│         └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键技术决策

1. **多实例部署策略**：

- 每个实例绑定不同的端口和NPU设备
- 使用环境变量`ASCEND_RT_VISIBLE_DEVICES`控制NPU可见性
- 每个实例独立进程，避免GIL限制

2. **客户端并发模型**：

- 使用asyncio + aiohttp实现异步并发请求
- 支持批量文件打包上传
- 实现滑动窗口限流控制

3. **资源监控方案**：

- CPU/内存：psutil进程级监控
- NPU显存：torch_npu相关API
- 采用独立监控线程，定时采样

4. **网格搜索实现**：

- 参数组合笛卡尔积生成
- 支持断点续测
- 每组参数独立测试周期

### 目录结构

```
mineru_perf_tool/
├── __init__.py
├── cli.py                    # [NEW] CLI入口，命令行参数解析
├── config.py                 # [NEW] 配置管理，默认配置和YAML加载
├── server/
│   ├── __init__.py
│   ├── manager.py            # [NEW] 服务端管理器，多实例启停控制
│   └── instance.py           # [NEW] 单实例进程封装
├── client/
│   ├── __init__.py
│   ├── benchmark.py          # [NEW] 压测客户端，异步请求发送
│   └── batch.py              # [NEW] 文件批处理，组批逻辑
├── monitor/
│   ├── __init__.py
│   ├── resource.py           # [NEW] 资源监控器，CPU/内存/GPU/NPU
│   └── collector.py          # [NEW] 数据采集器，定时采样
├── evaluator/
│   ├── __init__.py
│   ├── accuracy.py           # [NEW] 精度评测，集成OmniDocBench
│   └── throughput.py         # [NEW] 吞吐量指标计算
├── optimizer/
│   ├── __init__.py
│   ├── grid_search.py        # [NEW] 网格搜索，参数组合遍历
│   └── result.py             # [NEW] 结果聚合和分析
├── report/
│   ├── __init__.py
│   ├── generator.py          # [NEW] 报告生成，JSON/CSV输出
│   └── visualization.py      # [NEW] 可视化，热力图等图表
├── utils/
│   ├── __init__.py
│   ├── file_utils.py         # [NEW] 文件操作工具
│   └── npu_utils.py          # [NEW] NPU设备工具函数
└── configs/
    ├── default.yaml          # [NEW] 默认配置模板
    └── omnidocbench.yaml     # [NEW] OmniDocBench模式配置
```

## 实现细节

### CLI命令设计

```
# 基本用法
mineru-perf run --input-dir ./data --devices 0,1 --backend hybrid-auto-engine

# 网格搜索
mineru-perf grid-search --input-dir ./data \
    --devices-range 1,2,4 \
    --instances-per-device-range 1,2 \
    --concurrency-range 10,20,50 \
    --batch-size-range 1,5,10

# OmniDocBench精度评测
mineru-perf run --mode omnidocbench \
    --omnidocbench-path ./OmniDocBench \
    --devices 0,1
```

### 核心数据结构

- `BenchmarkConfig`: 压测配置（设备、实例数、并发数等）
- `TestResult`: 单次测试结果（QPS、延迟分布、资源占用）
- `GridSearchResult`: 网格搜索结果聚合
- `ResourceSnapshot`: 资源快照（时间戳、CPU、内存、显存）

### 性能考虑

- 异步IO避免阻塞，提高并发效率
- 资源监控采样间隔可配置（默认1秒）
- 大文件上传使用流式传输
- 结果聚合使用增量计算，避免内存溢出

## Agent Extensions

### SubAgent

- **code-explorer**
- Purpose: 深入探索MinerU和OmniDocBench的API接口和数据结构
- Expected outcome: 获取准确的函数签名、参数类型和调用方式，确保集成代码的正确性