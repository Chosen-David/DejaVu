# 稀疏感知的FFN-TP优化（FlashFFN+求解器设计）

# 摘要


# 介绍

现有的LLM云端推理中，多 GPU 推理通常依赖张量并行（Tensor Parallel, TP）把单层线性计算拆分到多卡完成；此外现有云端推理。然而他具有两个局限性：
1. 目前的TP切分都是均匀的切分神经元权重[ref:Megatron、DeepSpeed TP都是机械的切的]，没有很好的考虑到激活的稀疏性和偏斜性[ref:The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers文章证明了稀疏是模型结构固有属性]。这也就造成了部分GPU上的神经元可能经常被激活而其余上的神经元基本不被激活，也即造成了不同GPU上的负载不均衡。此外有些神经元总是容易一起被激活[ref:共激活现象]，对于此类神经元如果将其放在不同的GPU会引入额外的通信开销。对于该问题，已经有相关预测器的工作可以预测出哪些神经元是可以被激活的，哪些是几乎不被激活的[ref:DEJAVU、Powerinfer]，哪些神经元往往是可以一起被共激活的。
2. 对于LLM推理中的FFN层，有热神经元的部分和冷神经元的部分，当我们将热神经元和冷神经元都切分到同一个GPU中时。由于神经元激活的偏斜性。缺少一个算子可以有效利用好GPU硬件资源的TensorCore和CUDACore来实现硬件层面的负载均衡
3. TP的通信开销往往是整个过程中最大的开销，也即对于计算得到结果后进行GPU间的通信。然而传统的通信模型往往基于Dense激活假设，也即所有的神经元都会被激活。他们只进行了通信和计算上的overlap优化[ref:Alpa: Automating Inter- and Intra-Operator Parallelism]没有利用上激活的偏斜性和稀疏性，由于NCCL往往需要固定的Tensor，并且构建最后要传输的数值的索引往往会引入额外的开销。因此现有的求解器往往仍然是基于Dense激活[ref:Oases]。



因此我们提出了一种神经元稀疏感知的云端 LLM 推理框架xxx，它可以对于将被处理的FFN权重感知激活稀疏，并且提供一种能够兼顾GPU间的load balance以及GPU内部TC和CC的load balance。从而实现对于云端推理效率的显著提升实验证明提升了xxxx

更具体的：

设计了一种TP划分策略

# 背景

**激活稀疏与预测器。** “Lazy Neuron Phenomenon” 从现象与实证角度说明 Transformer MLP（尤其 ReLU 类激活）存在高比例稀疏激活，且模型越大越稀疏，为“只计算/只通信有效部分”提供了理论与经验动机[1]。DejaVu 进一步提出在线预测上下文稀疏，并通过异步与硬件感知实现系统级加速[2]；PowerInfer 则利用幂律激活局部性与“热/冷神经元”思想提升单机推理效率[3]。  

**多 GPU 推理与 TP 基线。** Megatron-LM 代表了早期且影响广泛的层内模型并行（intra-layer model parallel）/TP 思路，并依赖集体通信聚合结果[4]；DeepSpeed Inference 针对推理端提出多项系统级优化并支持多 GPU 的高效推理执行[5]。这些系统通常默认“按维度均匀切分”，与用户草稿指出的“机械切分忽略稀疏与偏斜”一致。  

**计算-通信重叠与流水调度。** TokenWeave 通过把 tokens 划分为两个近似等分的子集实现粗粒度 compute-communication overlap，并进一步融合 AllReduce–RMSNorm 内核以减少 SM 争用[6]。[Oases] 关注张量模型并行训练中的重叠与自动规划（planner），强调对重叠代价建模与搜索的重要性[7]。AutoOverlap 把“chunk”作为抽象，支持更细粒度的融合与调度，从编译与运行时角度降低重叠实现门槛[8]。TileLink/Flux/FlashOverlap 等工作从 kernel fusion、tile-centric primitive、信号机制、以及 CTA 级流式方案等角度系统性推进“更细粒度的 overlap”。  


# 设计


## 问题建模
也即给定了预测器的预测结果为布尔矩阵M[S*F]，其中位置(i, j)代表第i个token是否激活了第j个神经元；
给定了token矩阵X[S*H]
给定了预测器掩码矩阵M[S*F]
给定了第一权重矩阵A[H*F]
给定了第二权重矩阵B[F*H]
已知TP到n个GPU上
如何设计TP策略，将A划分到A_1, A_2,...., A_n；将B划分到B_1, B_2, ...., B_n使得总的开销最小。
此外由于A存在HANS和LANS两种神经元，对于每个A_i会考虑分配多少HANS以及分配多少LANS。
更具体的，对于HANS中，存在共激活的现象。也即其实切分时候A中视为存在三种神经元，共激活HANS_GROUP、非共激活HANS_GROUP、LANS。
对于这三种分别均分n份到n个GPU上。也即每个GPU上会存在：
A_i={value:[也即A_i重排为Dense后的矩阵]， HANS_ID{HANS的索引位置}， LANS_ID{LANS的索引位置}}
拥有了这些我们可以同样对于B_i进行切分，对于B_i在知道HANS_ID和LANS_ID之后会重排成Dense版本的B_i
B_i={value:[也即B_i重排为Dense后的矩阵]， HANS_ID{HANS的索引位置}， LANS_ID{LANS的索引位置}}

划分好之后，对于token矩阵X和预测掩码矩阵M按照S维度进行chunked成chunk_0, ...., chunk_m多份,按照流水Buffer输入进行计算：
$ sigma(chunk_i @ A_i) @ B_i  $ -> $ AllReduce(chunk_i) $


这样便可以实现chunk_0已经算完在进行Ring AllReduce通信的时候，chunk_1可以直接计算实现计算和通信的overlap。
此外，如何划分好HANS和LANS使得TensorCore和CUDACore的运行开销几乎相同？GPU在这种只计算稀疏有效部分然后传输结果的有效值部分以及对应索引的时候的通信量如何建模？由于对于X进行了chunk，如何保证切分后每个GPU的LANS部分在不同chunk运行时候参考的LANS索引在chunk后的M的运算参与是不是基本都可以有效？

## 设计总览

如图：
![](./figs/flow.png)

对于神经元，先通过离线求解器求解出每个GPU放哪些神经元以及具体到每个GPU的TC和CC会用哪些神经元执行。
对于X拆分成两个chunk，流水输入p个GPU。并且让计算和通信OverLap
对于通信，只传输有效激活的量以及对应的索引位置
对于计算，HANS激活的使用TC，LANS激活的使用CC。并且将整个FFN层融合成一个fused Kernel

## 优化求解器设计


### 第一层求解器：神经元->GPU

由于部分神经元总是经常高频率甚至共同被激活，因此第一层求解器也即如何将高频率、共激活的神经元作为高共激活神经元组HCNS；其余不常激活的定义为低共激活神经元组LCNS

首先，基于掩码矩阵得到共激活矩阵 $CW$：

$$
w_{uv}
= \frac{1}{S}\sum_{i=1}^{S} M_{i,u}M_{i,v}
$$

其中w_{uv}代表共激活权重，它代表了神经元对之间的共激活强度：

之后，通过谱聚类将这些权重自动分成两类，一类是高激活的HCNS，其余的部分作为LCNS。

进一步的，将HCNS和LCNS分别均匀的分到p个GPU上，也即第一层求解器的划分：神经元->GPU。最终得到rule1_i={HCNS_i, LCNS_i} 。比如[0~32]，其中奇数是HCNS、偶数为LCNS。则其中GPU0: rule1_0={HCNS:[1,3]， LCNS:[0,2]}。从而实现GPU间的负载均衡。

由于掩码矩阵包含在了预测器内部，这种开销是离线的开销。




### 第二层求解器：GPU内部神经元->TC、CC


为了更好的利用好GPU内部的TC和CC资源，第二层求解器设计为如何把之前划分好的GPU内部的神经元->TC、CC上面

目标：

$$
\min
\left|
T_{TC,i} - T_{CC,i}
\right|
$$


这一步可以通过启发式感知的方法，因为在线计算场景下的开销用时和离线计算开销一样是相对的，也即离线计算用时最低的配置的往往也在在线计算用时表现最好，因此该层求解器通过遍历神经元维度去计算相对用时最低的配置。rule2_i={HANS_i, LANS_i}，从而实现GPU内部的负载均衡

#### 第三层求解器：Token->流水计算

在真正运行的时候，X矩阵可以被划分成两个Chunk进行通信和计算的Overlap。Chunk 调度本质为流水线调度问题：

因为更细粒度的建模开销反而很大，因此这里粗粒度的划分成两个均等的子集进行OverLap



## FlashFFN的混合Kernel
计算部分：



针对整个激活稀疏的FFN设计了一个fused Kernel：FlashFFN进行加速
HANS部分：
对于M[S*H] @ M[H*F_i] @ M[F_i*H]，在TensorCore上
LANS部分：
对于M[Sc*H] @ V1[H*1] @ V2[1*H]的MVV运算，在CUDACore上面

计划用类似FlashAttention的方法进行加速

更具体的，首先依据求解器得到的神经元->GPU上的映射，将A取出相关的索引重排为Dense形式到i号GPU上也即A_i,B同理得到B_i
也即
1. GPU层级上，每个GPU上会存在权重值：
A_i={value:[也即A_i重排为Dense后的矩阵]，shape:[H*(F_i^TC+F_i^CC)], HANS_ID{HANS的索引位置}， LANS_ID{LANS的索引位置}}
拥有了这些我们可以同样对于B_i进行切分，对于B_i在知道HANS_ID和LANS_ID之后会重排成Dense版本的B_i
B_i={value:[也即B_i重排为Dense后的矩阵]，shape:[(F_i^TC+F_i^CC)*H], HANS_ID{HANS的索引位置}， LANS_ID{LANS的索引位置}}
同时也存在
2. GPU内部Kernel层级上会存在任务队列：
Q_TC = {TC_TILE0, TC_TILE1, \dots, }
Q_CC = {CC_TILE0, CC_TILE1, \dots, }
以及
3. 输入X的输入层级上：
$chunk_k={value:[也即X chunk的部分重排为Dense后的矩阵]， TOKEN_ID{也即这些chunk的token原本在X中的位置}}$
根据求解器测出的最佳Tiling配置，每个SM上TensorCore会启动 $num_TC$ 的Block，CUDACore会启动 $num_CC$ 的Block。
这些常驻Block会不断的从任务队列中取出任务进行计算。

更具体的，chunk_k取出Q_TC中的TC_TILEi之后，会根据该TILE从A_i和B_i中取出相关的权重值重排成Dense进行FlashMMM计算，同理取出Q_CC后便进行FlashMVV运算。而FlashMMM是在TC上运行的，FlashMVV是在CC上运行的

## 稀疏通信

之前的研究默认是全量通信，此处我们基于稀疏激活预测器，可以只传输有效激活的部分，以及这些GPU上激活的索引id。

传输完之后进行定制化的AllGather重新聚集到结果矩阵里面

## 实验

待更新，