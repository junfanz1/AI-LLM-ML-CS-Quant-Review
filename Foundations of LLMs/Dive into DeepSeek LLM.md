Dive into DeepSeek LLM, by Xiaojing Ding, 2025 

<img src="https://github.com/user-attachments/assets/2f24506f-a460-40bd-b3d2-c113f3ef9143" width="32%" height="32%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. LLM Intro](#1-llm-intro)
- [2. DL & RL](#2-dl-rl)
- [3. NLP](#3-nlp)
- [4. RL & DeepSeek-R1-Zero](#4-rl-deepseek-r1-zero)
- [5. Cold-Start RL & DeepSeek-R1](#5-cold-start-rl-deepseek-r1)
- [6. DeepSeek-R1 Architecture](#6-deepseek-r1-architecture)
- [7. DeepSeek-R1 Training ](#7-deepseek-r1-training)
- [8. DeepSeek-R1 Development](#8-deepseek-r1-development)
- [9. DeepSeek-R1 Development 2](#9-deepseek-r1-development-2)
- [10. FIM, Context Cache](#10-fim-context-cache)
- [11. Backend Business Code Generation](#11-backend-business-code-generation)
- [12. DeepSeek-R1 & V3: SaaS Recommendation System](#12-deepseek-r1-v3-saas-recommendation-system)

<!-- TOC end -->

<!-- TOC --><a name="1-llm-intro"></a>
## 1. LLM Intro

GPT = 多个Transformer Blocks（=Multi-Head Self Attention + Feed Forward Network）
- 多头自注意力：每个注意力头包括matmul, mask, softmax, dropout，防止过拟合、确保信息有效聚合
- 前馈神经网络 = 两层线性变换 + GELU激活函数，进一步提取特征、提高模型表达能力。层归一化和残差连接在每个子层间稳定梯度、加速收敛。

并行
- 数据并行：训练数据切分到不同设备计算，每个设备计算梯度后用参数同步（All-Reduce）更新参数。
- 模型并行：模型切分到不同设备，包括Layer-wise Parallelism（不同层分到不同设备，适用深层网络）和Tensor Parallelism（每一层参数切分到多个设备，适用大规模权重矩阵）
- 混合并行：浅层采用数据并行，深层采用模型并行
- 流水线并行：把模型按阶段切分到不同设备，提高硬件利用率、减少空闲时间。要结合Micro-batching和负载均衡来优化。

模型初始化：防止梯度消失爆炸，Xavier/He初始化；DeepSeek-R1用动态权重初始化+RL，确保模型稳定收敛；DeepSeek-V3用MoE+分布式权重初始化，优化跨节点参数一致性。

PEFT：DeepSeek-R1用LoRA + Adapter微调；DeepSeek-V3用Adapter + Prefix Tuning + MoE。

推理优化
- 量化
  - 后训练量化Post-Training Quantization：快速模型部署和推理加速，不用重新训练，可能出现精度下降。
  - 量化感知训练Quantization Aware Training：在模型训练阶段就量化，在前向传播模拟量化Fake Quantization（用伪量化层模拟低精度计算）操作、反向传播仍用浮点数计算梯度。训练完后，模型可导出为低量化模型，供推理部署适用。适合精度敏感任务（语音识别图像分类），但训练成本高。
  - Adaptive Quantization, Mixed-Precision Inference, Hardware-Aware Quantization
- 剪枝（模型稀疏化）：Weight pruning去除近0权重；Structured pruning以卷积核、通道、层为单位剪枝，适合硬件加速
- 知识蒸馏
  - Soft targets通过教师的概率分布（软标签或中间层特征）指导学生；Feature Distillation不仅关注输出层，还对中间层特征表示进行蒸馏，更好理解数据内部结构。
  - Distillation loss = soft label loss（学生与老师差异，用温度调整的交叉熵）+ hard label loss（学生预测结果与真实标签差异，确保模型有基础分类能力）
- DeepSeek-R1：动态量化 + 自适应剪枝；DeepSeek-V3：MoE + 结构化剪枝、量化感知训练QAT + 跨模态知识蒸馏

性能优化
- 模型结构：Sparsity（引入稀疏连接，减少计算，保留关键路径）；MoE（动态路由分配到子模型，只激活部分参数，减少计算）；轻量架构Lightweight Architecture（精简Transformer变体（MobileBERT, DistilBERT）减少模型层数和参数规模）
- 硬件加速：量化、FP16与混合精度推理
- 并行计算：模型并行、流水线并行、批量推理Batch Inference适用高并发场景、异步推理Asynchronous Inference适用多任务环境（减少任务间阻塞）
- 低延迟：动态推理路径Dynamic Inference Pathways（基于输入数据复杂度动态调整推理路径，DeepSeek-V3动态路由仅激活相关的子模型）；模型裁剪Model Pruning（边缘设备上部署裁剪与量化的模型，减少开销）；缓存机制（DeepSeek-V3用KV缓存，多轮对话和连续推理中复用历史计算结果）

<!-- TOC --><a name="2-dl-rl"></a>
## 2. DL & RL

优化器
- 随机梯度下降SGD：受限于收敛速度和震荡问题，适用小模型小数据
- Adam (Adaptive Momentum Estimation)：自动调整学习率（无需手动调节），更新参数时同时考虑一阶矩（均值）和二阶矩（方差），自适应调整每个参数的学习率。
- LAMB (Layer-wise Adaptive Moments optimizer for Batch training）：适用大规模分布式训练，= Adam + 层级自适应学习率调整，保持稳定性同时扩展批量大小，提高分布式训练效率，对不同层的参数自适应缩放。

PyTorch
- 反向传播：计算图Computation Graph描述神经网络数据流动和运算过程，前向传播是图的自上而下计算，反向传播是图的逆向遍历（链式法则计算梯度）。PyTorch动态计算图，在运行时建图，灵活性高；TensorFlow是静态计算图，在模型定义阶段建图，适合大规模分布式训练。
- Autograd自动求导：自动计算梯度，简化反向传播算法实现。

学习率
- 太小导致训练速度慢、局部最优；太大导致无法收敛，在最优解附近震荡。
- 学习率Decay：动态调整学习率，有Step Decay, Exponential Decay, Cosine Annealing（先快速下降后缓慢收敛，适用训练周期长的大模型）, Adaptive Decay（根据在验证集上的性能调整，性能不提升时降低学习率）
- 学习率Warm-up：参数不稳定时，大学习率会导致不稳定、损失爆炸，预热侧率可以初期稳定收敛，避免大梯度导致训练不稳定。

强化学习
- Q-Learning：基于时间差分Temporal Difference（动态规划+蒙特卡洛方法，估计Agent与环境交互获得的状态值，不需要等完整回合结束再更新策略，而是实时更新）的RL，寻找最优动作。随着学习不断进行，Q值会逼近最优状态，Agent可在任何状态做出最优决策。
- Deep Q-Network（DQN）：用深度神经网络作为函数逼近器来估计Q值函数，解决了Q-Learning在高位状态空间的维度灾难问题。但在连续动作空间和策略优化方面存在局限（策略不稳定、样本效率低），因此引入策略梯度方法（Actor-Critic）来优化策略函数，使其高效决策并且收敛。
  - Experience Replay经验回放：Agent环境交互的状态、动作、奖励、下一状态、终止标志的交互数据以五元组存储到Replay Buffer缓冲区，每次模型参数更新都从缓冲区Replay Buffer随机采样mini-batch历史经验进行训练，可以降低样本间相关性，提高样本利用率，让数据分布更iid，提高泛化能力和稳定性。
  - 目标网络Target Network：参数固定更新Hard Update（每隔几千步与主Q网络参数完全同步，可以减少快速参数更新的Q值震荡，但学习效率低）、软更新Soft Update（指数加权逐步调整参数，缓慢向主网络靠拢）、多目标网络Multi Target Networks、动态调整同步频率Dynamic Synchronization Frequency Adjustment（根据模型学习进展动态调整目标网络的更新频率）。
  - Double DQN：缓解Q值的高估偏差，解耦动作选择和价值估计两个过程，缓解过度乐观。
  - Dueling DQN：把Q值分解两部分（状态价值函数V + 优势函数A），共享前置特征提取层，适合状态空间大、冗余信息多的游戏场景，即使动作选择对最终奖励影响不大，也能有效评估状态价值。
- Actor-Critic：Actor生成动作策略，Critic估计状态-动作值函数，策略与价值分离并协同优化，用DQN价值评估能力和策略梯度优化能力，高效稳定学习。
  - Advantage Function衡量当前动作相对于平均水平的优势成都，帮Actor更精准调整策略，防止单一的即时奖励波动产生不必要的策略更新。
  - 变体：A2C (Advantage Actor-Critic)、A3C (Asynchronous Advantage Actor-Critic)，引入异步并行更新机制和改进的优势估计策略，提高训练效率和泛化能力。
- Multi-Agent RL (MARL)：每个智能体还要考虑其他智能体行为对环境和自身的奖励。挑战：非平稳环境、信用分配、协作与竞争平衡。


<!-- TOC --><a name="3-nlp"></a>
## 3. NLP

<!-- TOC --><a name="4-rl-deepseek-r1-zero"></a>
## 4. RL & DeepSeek-R1-Zero

<!-- TOC --><a name="5-cold-start-rl-deepseek-r1"></a>
## 5. Cold-Start RL & DeepSeek-R1

<!-- TOC --><a name="6-deepseek-r1-architecture"></a>
## 6. DeepSeek-R1 Architecture

<!-- TOC --><a name="7-deepseek-r1-training"></a>
## 7. DeepSeek-R1 Training 

<!-- TOC --><a name="8-deepseek-r1-development"></a>
## 8. DeepSeek-R1 Development

<!-- TOC --><a name="9-deepseek-r1-development-2"></a>
## 9. DeepSeek-R1 Development 2

<!-- TOC --><a name="10-fim-context-cache"></a>
## 10. FIM, Context Cache

<!-- TOC --><a name="11-backend-business-code-generation"></a>
## 11. Backend Business Code Generation

<!-- TOC --><a name="12-deepseek-r1-v3-saas-recommendation-system"></a>
## 12. DeepSeek-R1 & V3: SaaS Recommendation System
