# Quick Start Guide - Sparse TP FFN

## 快速开始

### 1. 环境准备

```bash
# 安装依赖
pip install torch numpy scipy scikit-learn

# 如果要使用Triton kernels
pip install triton

# 验证环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. 运行测试

```bash
cd /Users/wangyuanshuo/workspace/SA-TP-FFN-Resorver/DejaVu/Decentralized_FM_alpha

# 运行单元测试
python -m tests.test_sparse_tp_ffn

# 运行端到端集成测试
python -m tests.test_e2e_integration

# 或使用脚本一键运行
./scripts/run_e2e_test.sh
```

### 3. 基本使用

#### 离线分析模式

```python
from runtime import create_sparse_tp_ffn
import numpy as np

# 创建模型
ffn = create_sparse_tp_ffn(
    hidden_dim=4096,
    intermediate_dim=16384,
    num_gpus=8,
    rank=0,  # 当前GPU rank
    activation='gelu',
)

# 准备样本数据(从预测器收集)
sample_masks = np.random.random((500, 16384)) > 0.5

# 运行离线分析
results = ffn.offline_analysis(sample_masks)
print(f"Analysis time: {results['analysis_time_s']:.2f}s")
print(f"HANS neurons: {results['total_hans']}")
print(f"LANS neurons: {results['total_lans']}")

# 保存分析结果(自动保存到 ./analysis_results/)
```

#### 在线推理

```python
import torch

# 加载模型和已保存的分析结果
ffn = create_sparse_tp_ffn(
    hidden_dim=4096,
    intermediate_dim=16384,
    num_gpus=8,
    rank=0,
)

# 自动加载分析结果
# 或手动: ffn.offline_analysis(load_cached=True)

# 准备输入
x = torch.randn(2048, 4096, device='cuda', dtype=torch.float16)
mask = torch.rand(2048, 16384, device='cuda') > 0.5

# 运行推理
output = ffn(x.cuda(), mask.float().cuda())
```

### 4. 与DejaVu集成

```python
from integration import ParallelSparseTPMLP, convert_dejavu_mlp_to_sparse_tp
from transformers import AutoModel

# 方法1: 直接创建
model = ParallelSparseTPMLP(
    hidden_dim=4096,
    intermediate_dim=16384,
    layer_idx=0,
    process_group=process_group,
    model_name="my_model",
)

# 方法2: 转换现有的DejaVu模型
# original_mlp = ...  # ParallelFusedMLPDejavu实例
# new_mlp = convert_dejavu_mlp_to_sparse_tp(original_mlp, layer_idx=0, model_name="my_model")
```

### 5. 分析结果存储

离线分析的结果自动存储在以下结构中:

```
analysis_results/
├── model_name/
│   ├── config.json                    # 模型配置
│   ├── layer_0/
│   │   ├── result.json                # 分析结果元数据
│   │   ├── coactivation_matrix.npy    # 共激活矩阵
│   │   └── activation_frequencies.npy # 激活频率
│   ├── layer_1/
│   │   └── ...
└── index.json                         # 全局索引
```

#### 管理存储结果

```python
from solver import AnalysisResultStore

store = AnalysisResultStore("./analysis_results")

# 列出所有模型
models = store.list_models()

# 列出模型的所有层
layers = store.list_layers("my_model")

# 获取分析结果摘要
summary = store.get_result_summary("my_model", layer_id=0)

# 加载完整结果
result = store.load_result("my_model", layer_id=0, load_matrices=True)

# 删除结果
store.delete_result("my_model", layer_id=0)
```

### 6. 使用求解器组件

如果需要单独使用各个求解器:

```python
from solver import (
    CoactivationAnalyzer,
    NeuronPartitioner,
    GPUBalancer,
    TCCCBalancer,
    PipelineScheduler,
)

# 第一层: 共激活分析
analyzer = CoactivationAnalyzer(num_neurons=16384)
analyzer.update(masks)
coactivation_weights = analyzer.compute_coactivation_weights()

# 第一层: 神经元分组
partitioner = NeuronPartitioner(num_neurons=16384, num_gpus=8)
partitions = partitioner.analyze_and_partition(masks, method='spectral')

# 第二层: GPU负载均衡
balancer = GPUBalancer(num_gpus=8, hidden_dim=4096, intermediate_dim=16384)
balanced, info = balancer.balance_partitions(partitions, coactivation_weights, frequencies)

# 第二层: TC/CC负载均衡
tc_cc = TCCCBalancer(hidden_dim=4096, intermediate_dim=16384)
assignment = tc_cc.balance(hans_idx, lans_idx, frequencies, seq_len=2048)

# 第三层: 流水线调度
scheduler = PipelineScheduler(num_gpus=8, hidden_dim=4096)
schedule = scheduler.schedule(seq_len=2048, activation_sparsity=0.5)
```

### 7. 性能调优

#### TC/CC平衡参数

```python
tc_cc_balancer = TCCCBalancer(
    hidden_dim=4096,
    intermediate_dim=16384,
    tc_tflops=312.0,      # A100 TensorCore TFLOPS
    cc_tflops=19.5,       # A100 CUDACore TFLOPS
    tc_efficiency=0.8,    # TensorCore效率
    cc_efficiency=0.6,    # CUDACore效率
)
```

#### 流水线调度参数

```python
scheduler = PipelineScheduler(
    num_gpus=8,
    hidden_dim=4096,
    target_chunk_size=512,  # 目标chunk大小
    max_chunks=8,           # 最大chunk数
)
```

#### 通信优化参数

```python
from communication import create_sparse_communicator

communicator = create_sparse_communicator(
    hidden_dim=4096,
    num_gpus=8,
    rank=0,
    mode='adaptive',
    sparsity_threshold=0.5,  # 稀疏度阈值
    compression_ratio=0.3,   # 预期压缩比
)
```

### 8. 完整示例

参见:
- `examples/example_usage.py` - 基本使用示例
- `tests/test_e2e_integration.py` - 完整集成测试

## 故障排除

### Q: Triton kernel编译失败
A: 确保安装了正确版本的triton,或使用标准PyTorch实现作为fallback

### Q: 分布式初始化失败
A: 检查NCCL安装和CUDA版本兼容性

### Q: 离线分析结果不匹配
A: 使用`force_reanalyze=True`重新分析,或删除旧结果

### Q: 性能不如预期
A: 
1. 检查TC/CC分配是否平衡
2. 调整流水线chunk数量
3. 验证通信稀疏度配置

## 下一步

1. 在您的模型上运行离线分析
2. 对比sparse TP与传统TP的性能
3. 根据实验结果调优参数
4. 查看ARCHITECTURE.md了解详细设计
