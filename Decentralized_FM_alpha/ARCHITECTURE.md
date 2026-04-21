# Sparse-Aware FFN Tensor Parallel System - Architecture Documentation

## 项目概述

本项目实现了一个稀疏感知的FFN张量并行优化系统,针对大语言模型云端推理中的FFN层进行优化。系统通过三层求解器实现神经元级别的负载均衡,利用混合TensorCore/CUDACore执行,并通过流水线调度实现计算通信重叠。

## 核心创新点

### 1. 稀疏感知的TP划分
- 传统的TP切分均匀切分神经元权重,忽略了激活的稀疏性和偏斜性
- 本系统根据预测器结果识别HANS(高激活神经元组)和LANS(低激活神经元组)
- 考虑神经元共激活现象,将共激活神经元分配到同一GPU以减少通信开销

### 2. 混合TC/CC执行
- HANS神经元在TensorCore上以密集计算方式执行
- LANS神经元在CUDACore上以稀疏计算方式执行
- 通过求解器动态调整分配,实现TC和CC的负载均衡

### 3. 流水线调度优化
- 将Token序列分chunk进行流水线处理
- 实现计算和AllReduce通信的重叠
- 稀疏通信只传输有效激活部分

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        SparseTPFFN (主模块)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────── 离线分析阶段 ───────────────────┐        │
│  │                                                   │        │
│  │  ┌──────────────────────────────────────────┐   │        │
│  │  │  第一层求解器: 神经元分组                   │   │        │
│  │  │  - CoactivationAnalyzer (共激活分析)      │   │        │
│  │  │  - NeuronPartitioner (HANS/LANS划分)      │   │        │
│  │  └──────────────────────────────────────────┘   │        │
│  │                    ↓                            │        │
│  │  ┌──────────────────────────────────────────┐   │        │
│  │  │  第二层求解器: 负载均衡                     │   │        │
│  │  │  - GPUBalancer (GPU间负载均衡)            │   │        │
│  │  │  - TCCCBalancer (TC/CC负载均衡)           │   │        │
│  │  └──────────────────────────────────────────┘   │        │
│  │                    ↓                            │        │
│  │  权重重排: 按TC/CC分配重排权重矩阵              │        │
│  └──────────────────────────────────────────────────┘        │
│                                                                 │
│  ┌─────────────────── 在线推理阶段 ───────────────────┐        │
│  │                                                   │        │
│  │  输入: X[S, H] + 预测掩码 M[S, F]                │        │
│  │                    ↓                            │        │
│  │  ┌──────────────────────────────────────────┐   │        │
│  │  │  第三层求解器: 流水线调度                   │   │        │
│  │  │  - PipelineScheduler (Chunk划分)          │   │        │
│  │  └──────────────────────────────────────────┘   │        │
│  │                    ↓                            │        │
│  │  ┌──────────────────────────────────────────┐   │        │
│  │  │  FlashFFN Kernel (混合执行)                │   │        │
│  │  │  - TC: HANS密集计算                       │   │        │
│  │  │  - CC: LANS稀疏计算                       │   │        │
│  │  └──────────────────────────────────────────┘   │        │
│  │                    ↓                            │        │
│  │  ┌──────────────────────────────────────────┐   │        │
│  │  │  Sparse Communication (稀疏通信)           │   │        │
│  │  │  - SparseAllReduce (只传有效值+索引)      │   │        │
│  │  └──────────────────────────────────────────┘   │        │
│  │                    ↓                            │        │
│  │  输出: Y[S, H]                                  │        │
│  └──────────────────────────────────────────────────┘        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 模块详细说明

### 1. Solver模块 (`solver/`)

#### 1.1 第一层求解器: 神经元分组

**CoactivationAnalyzer** (`coactivation_analyzer.py`)
- 分析神经元共激活模式
- 计算共激活权重矩阵 CW[F,F]
- 支持在线更新和离线分析
- 提供谱聚类和层次聚类方法

```python
# 示例
analyzer = CoactivationAnalyzer(num_neurons=16384)
analyzer.update(activation_masks)  # 更新统计信息
CW = analyzer.compute_coactivation_weights()  # 获取共激活矩阵
```

**NeuronPartitioner** (`neuron_partitioner.py`)
- 将神经元划分为HANS和LANS
- 基于激活频率和共激活模式
- 支持三种划分策略: even, proportional, adaptive
- 输出每个GPU的神经元分配方案

```python
# 示例
partitioner = NeuronPartitioner(num_neurons=16384, num_gpus=8)
partitions = partitioner.analyze_and_partition(masks, method='spectral')
# partitions[0] = NeuronPartition(hans_indices=[...], lans_indices=[...])
```

#### 1.2 第二层求解器: 负载均衡

**GPUBalancer** (`gpu_balancer.py`)
- GPU间负载均衡
- 考虑计算时间、内存使用和通信开销
- 支持三种策略: compute_aware, memory_aware, hybrid

```python
# 示例
balancer = GPUBalancer(num_gpus=8, hidden_dim=4096, intermediate_dim=16384)
balanced_partitions, info = balancer.balance_partitions(
    partitions, coactivation_weights, activation_frequencies, strategy='hybrid'
)
```

**TCCCBalancer** (`tc_cc_balancer.py`)
- GPU内部TC/CC负载均衡
- 目标: min|T_TC - T_CC| 以最大化重叠效率
- 支持三种搜索方法: heuristic, exhaustive, binary

```python
# 示例
tc_cc_balancer = TCCCBalancer(hidden_dim=4096, intermediate_dim=16384)
assignment = tc_cc_balancer.balance(
    hans_indices, lans_indices, activation_frequencies, sequence_length
)
# assignment.tc_indices, assignment.cc_indices
```

#### 1.3 第三层求解器: 流水线调度

**PipelineScheduler** (`pipeline_scheduler.py`)
- Token chunk划分
- 计算和通信重叠调度
- 支持策略: two_chunk, adaptive, optimal

```python
# 示例
scheduler = PipelineScheduler(num_gpus=8, hidden_dim=4096)
schedule = scheduler.schedule(
    sequence_length=2048,
    activation_sparsity=0.5,
    strategy='two_chunk'
)
# schedule.chunks, schedule.total_time, schedule.overlap_efficiency
```

### 2. Kernels模块 (`kernels/`)

**FlashFFN** (`flash_ffn.py`)
- 融合稀疏FFN kernel
- Triton实现,支持TC和CC混合执行
- HANS部分: 密集矩阵乘法(TensorCore)
- LANS部分: 稀疏矩阵向量乘(CUDACore)

```python
# 示例
ffn = FlashFFN(
    hidden_dim=4096,
    intermediate_dim=16384,
    hans_indices=hans_idx,
    lans_indices=lans_idx,
    activation='gelu'
)
output = ffn(input_tensor, prediction_mask)
```

### 3. Communication模块 (`communication/`)

**SparseAllReduce** (`sparse_allreduce.py`)
- 稀疏AllReduce实现
- 只传输激活的神经元值和索引
- 支持自适应模式,根据稀疏度选择稠密或稀疏通信

```python
# 示例
communicator = create_sparse_communicator(
    hidden_dim=4096, num_gpus=8, rank=0, mode='adaptive'
)
result, work = communicator.adaptive_allreduce(tensor, mask, async_op=True)
```

### 4. Runtime模块 (`runtime/`)

**SparseTPFFN** (`sparse_tp_ffn.py`)
- 端到端集成模块
- 管理完整的离线分析和在线推理流程

```python
# 完整使用示例
from runtime import create_sparse_tp_ffn

# 创建实例
ffn = create_sparse_tp_ffn(
    hidden_dim=4096,
    intermediate_dim=16384,
    num_gpus=8,
    rank=0
)

# 离线分析
analysis_results = ffn.offline_analysis(sample_masks)

# 在线推理
output = ffn(input_tensor, prediction_mask)

# 获取统计信息
stats = ffn.get_statistics()
```

## 文件结构

```
Decentralized_FM_alpha/
├── __main__.py              # 主入口
├── solver/                  # 求解器模块
│   ├── __init__.py
│   ├── coactivation_analyzer.py    # 第一层: 共激活分析
│   ├── neuron_partitioner.py       # 第一层: 神经元分组
│   ├── gpu_balancer.py             # 第二层: GPU间均衡
│   ├── tc_cc_balancer.py           # 第二层: TC/CC均衡
│   └── pipeline_scheduler.py       # 第三层: 流水线调度
├── kernels/                 # 高性能Kernel
│   ├── __init__.py
│   └── flash_ffn.py               # FlashFFN Triton Kernel
├── communication/           # 通信模块
│   ├── __init__.py
│   └── sparse_allreduce.py         # 稀疏AllReduce
├── runtime/                 # 运行时系统
│   ├── __init__.py
│   └── sparse_tp_ffn.py           # 端到端集成
├── tests/                   # 测试
│   └── test_sparse_tp_ffn.py      # 端到端测试
└── examples/                # 示例
    ├── __init__.py
    └── example_usage.py            # 使用示例
```

## 使用流程

### 1. 安装依赖

```bash
pip install torch numpy scipy scikit-learn triton
```

### 2. 运行测试

```bash
python -m tests.test_sparse_tp_ffn
```

### 3. 运行示例

```bash
python -m examples.example_usage
```

### 4. 集成到现有系统

```python
from runtime import create_sparse_tp_ffn

# 在模型初始化时
self.ffn = create_sparse_tp_ffn(
    hidden_dim=config.hidden_dim,
    intermediate_dim=config.intermediate_dim,
    num_gpus=world_size,
    rank=rank
)

# 离线分析阶段(训练后或定期执行)
sample_masks = collect_prediction_masks(calibration_data)
self.ffn.offline_analysis(sample_masks)

# 推理时
output = self.ffn(hidden_states, prediction_mask)
```

## 性能优化建议

### 1. 离线分析
- 使用具有代表性的校准数据集
- sample_masks数量建议500-1000
- 定期重新分析以适应分布变化

### 2. TC/CC平衡
- 根据实际GPU型号调整tc_tflops和cc_tflops参数
- 对于不同序列长度可能需要不同的配置

### 3. 流水线调度
- 对于长序列(>2K),使用adaptive策略
- 对于短序列,使用two_chunk策略
- 调整chunk数量以平衡内存和效率

### 4. 通信优化
- 在高稀疏度(>70%)场景下效果最好
- 考虑网络拓扑和带宽配置参数

## 扩展性

### 支持新模型
系统设计为模型无关,只需提供:
1. 预测器接口(可选)
2. FFN权重维度

### 支持新硬件
1. 在TCCCBalancer中调整硬件参数
2. 在FlashFFN中优化kernel参数

### 自定义求解器
所有求解器都支持自定义策略:
- balance_strategy: 自定义负载均衡算法
- search_method: 自定义搜索方法

## 局限性与未来工作

### 当前局限
1. Triton kernel需要CUDA支持
2. 分布式测试需要多GPU环境
3. 预测器精度影响系统效果

### 未来工作
1. 支持更多GPU架构(AMD, Intel)
2. 自动调优系统
3. 与其他并行策略(PP, DP)集成
4. 端到端性能评估

## 引用

如果使用本系统,请引用相关论文:
- DejaVu: Contextual Sparsity for Efficient LLMs
- The Lazy Neuron Phenomenon
- PowerInfer

## 许可证

MIT License
