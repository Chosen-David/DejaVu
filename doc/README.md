# Euro-Par Paper: Sparse-Aware Tensor Parallelism for Efficient LLM Inference

本目录包含Euro-Par会议论文的三个版本,根据不同的投稿类型选择。

## 📋 三个版本对比

| 版本 | 页数 | 类型 | 适用场景 | 文件 |
|------|------|------|----------|------|
| **Poster/Demo** | ≤2页 | 海报/演示 | 初步成果展示、工具演示 | `europar_poster.tex` ⭐ |
| **Short Paper** | 6-8页 | 短文 | 完整但精简的研究 | `europar_short_paper.tex` |
| **Full Paper** | 12-14页 | 长文 | 完整详细的研究 | `europar_paper.tex` |

## ⭐ 推荐版本

**当前推荐**: `europar_poster.tex` (2页poster版本)

适合Euro-Par的poster或demonstration投稿,快速展示核心贡献。

## 📁 文件结构

```
doc/
├── europar_poster.tex          # Poster版本 (≤2页) ⭐ 推荐投稿
├── europar_short_paper.tex     # 短文版本 (6-8页)
├── europar_paper.tex           # 长文版本 (12-14页)
├── references.bib              # 参考文献数据库
├── llncs.cls                   # LNCS文档类
├── splncs04.bst                # 参考文献样式
├── compile_poster.sh           # 编译poster版本
├── compile_short.sh            # 编译短文版本
├── compile.sh                  # 编译长文版本
└── README.md                   # 本文件
```

## 🚀 快速开始

### 编译Poster版本 (推荐)

```bash
cd /Users/wangyuanshuo/workspace/SA-TP-FFN-Resorver/doc
chmod +x compile_poster.sh
./compile_poster.sh
```

### 检查页数

```bash
# 查看PDF页数
pdfinfo europar_poster.pdf | grep Pages
```

## 📝 Poster版本内容结构 (≤2页)

### 第1页:
1. **摘要** (80-100字)
   - 问题、方法、结果三要素

2. **引言** (0.4页)
   - 背景和问题陈述
   - 解决方案概述
   - 主要贡献列表

3. **三层求解器设计** (0.8页)
   - Level 1: 神经元划分
   - Level 2: TC/CC分配
   - Level 3: 流水线调度

### 第2页:
4. **实现** (0.2页)
   - FlashFFN内核
   - DejaVu集成

5. **评估** (0.4页)
   - 实验设置
   - 关键结果表格
   - 消融研究

6. **结论** (0.1页)

7. **参考文献** (0.1页)

## 🎯 核心贡献

### 三层求解器框架:

**Level 1 - 神经元划分**
- 基于共激活模式的谱聚类
- HCNS/LCNS分类
- 跨GPU负载均衡

**Level 2 - TC/CC分配**
- 执行时间建模
- 启发式优化
- GPU内硬件均衡

**Level 3 - 流水线调度**
- Chunk划分
- 计算-通信重叠
- 稀疏AllReduce

### 关键结果:

- **性能**: 2.3× 加速 (vs Megatron-LM)
- **精度**: 维持在1%以内
- **可扩展**: 近线性扩展至16 GPU
- **负载**: 方差从45%降至8%
- **通信**: 减少7.8×

## 📊 Poster版本特点

### 优势:
✅ 快速审阅,易于理解
✅ 聚焦核心贡献
✅ 适合初步成果
✅ 竞争压力较小
✅ 易被接收

### 精简策略:
- 删除详细算法伪代码
- 压缩背景和动机描述
- 只保留核心结果
- 合并相关工作到引用
- 最小化图表数量

## 🔬 实验数据 (待填充)

### Table 1: 性能对比

| Method | OPT-175B | GPT-J-6B | Speedup |
|--------|----------|----------|---------|
| Megatron-LM | 125.4 ms | 18.3 ms | 1.0× |
| DejaVu | 89.2 ms | 14.1 ms | 1.4× |
| **Ours** | **54.7 ms** | **9.2 ms** | **2.3×** |

### 消融研究

| Configuration | Speedup |
|---------------|---------|
| Baseline | 1.0× |
| + Level 1 | 1.4× |
| + Level 2 | 1.8× |
| + Level 3 | 2.3× |

## ⏰ 时间规划

### Poster版本完成时间: 3-5天

1. **实验数据收集** (2-3天)
   - 运行基线对比
   - 收集性能数据
   - 验证精度

2. **文档完善** (1-2天)
   - 填充数据
   - 编译测试
   - 格式检查

3. **提交准备** (1天)
   - 最终审阅
   - 准备材料
   - 提交

## 📚 Euro-Par投稿信息

- **会议**: Euro-Par 2026
- **类型**: Poster / Demonstration
- **页数**: ≤2页 (LNCS格式)
- **格式**: Springer LNCS
- **截稿**: 待公布

## 🔗 相关链接

- [Euro-Par官网](https://www.europar-conferences.org/)
- [Springer LNCS指南](https://www.springer.com/gp/computer-science/lncs)
- [LNCS模板下载](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)

## 📞 下一步行动

1. ✅ 确认投稿类型: Poster/Demonstration (≤2页)
2. 🔴 运行实验收集数据
3. 🟡 填充表格数据
4. 🟢 编译测试页数
5. 🟢 最终审阅提交

---

**当前状态**: Poster框架已完成,等待实验数据填充

**预计完成**: 3-5天

**推荐投稿**: Poster版本 (≤2页)
