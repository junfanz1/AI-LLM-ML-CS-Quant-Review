# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [1. Mixture of Experts (MoE)](#1-mixture-of-experts-moe)
   * [MoE模型架构](#moe)
   * [MoE优化](#moe-1)
   * [DeepSeek GRPO与LLM主流RLHF两大路线](#deepseek-grpollmrlhf)
   * [DeepSeek V-1 MoE, 2024.01 开山之作](#deepseek-v-1-moe-202401-)
   * [MoE核心原理](#moe-2)
- [2. DeepSeek](#2-deepseek)
   * [DeepSeek-V3](#deepseek-v3)
   * [DeepSeek-R1](#deepseek-r1)
   * [DeepSeek Janus](#deepseek-janus)
   * [Kimi-K1.5](#kimi-k15)
- [3. DeepSeek Open Source Week, 2025.02.24](#3-deepseek-open-source-week-20250224)
   * [Day 1. FlashMLA](#day-1-flashmla)
   * [Day 2. DeepEP](#day-2-deepep)
   * [Day 3. DeepGEMM](#day-3-deepgemm)
   * [Day 4. DualPipe](#day-4-dualpipe)
   * [Day 5. 3FS](#day-5-3fs)
- [Acknowledgements](#acknowledgements)

<!-- TOC end -->

<!-- TOC --><a name="1-mixture-of-experts-moe"></a>
# 1. Mixture of Experts (MoE)

Mixture of Experts (MoE) 混合专家模型。参数小、多专家，监督学习+分而治之，模块化神经网络的基础，像集成学习。根据scaling law大模型性能好，而推理时只执行部分参数，故DeepSeek成本低。

<!-- TOC --><a name="moe"></a>
## MoE模型架构
- 稀疏MoE层，代替了Transformer FFN层（节省算力），包含很多专家，每个专家是一个神经网络。稀疏性让部分专家被激活，非所有参数参与计算。高效推理同时，扩展到超大规模，提升模型表征能力。
- 专家模块化，不同专家学习不同特征，处理大数据。门控网络Gating Network或路由=可学习的门控网络+专家间负载均衡，动态协调哪些token激活哪些专家参与计算，与专家一起学习。稀疏门控激活部分专家，稠密门控激活所有专家，软门控合并专家与token并可微。

训练效率提升
- 训练。专家并行EP使用All2All通讯（带宽少），每个专家处理一部分batch（增加吞吐）。
- 推理。只激活少量专家（低延迟），增加专家数量（推理成本不变）。

<!-- TOC --><a name="moe-1"></a>
## MoE优化
- 专家并行计算
- 提高容量因子Capacity Factor和显存带宽
- MoE模型蒸馏回对应的稠密小模型
- 任务级别路由+专家聚合，简化模型减少专家

<!-- TOC --><a name="deepseek-grpollmrlhf"></a>
## DeepSeek GRPO与LLM主流RLHF两大路线
- On-Policy (PPO)：每次训练都基于自己的生成模型Actor，通过教练Critic反馈奖励。好：效率高，坏：模型能力低。PPO共有4个模型（Actor, Critic, Reward, Reference），计算大。
- Off-Policy (DPO)：基于现有标注进行分析，可能样本与模型不匹配。好：可能达到模型上限，坏：效率低。
- GRPO=无需价值函数，与奖励模型的比较性质对齐，KL惩罚在损失函数中。DeepSeek GRPO避免了PPO用Critic Value Model近似，而是用同一问题下多个采样输出的平均奖励作基线，这样Actor（没了Critic）直接去对齐Reward，求均值后再去跟Policy求KL散度。

<!-- TOC --><a name="deepseek-v-1-moe-202401-"></a>
## DeepSeek V-1 MoE, 2024.01 开山之作
- 细分了很多小专家，小而多，不同专家可以灵活组合（FFN参数减半，专家数量加倍）。
- 把共享专家隔离出来（单独多了一个FFN），让其他专家获得共同知识，提高专家的专业程度，部分专家在不同Token或层间共享参数，减少路由冗余。
- 负载均衡内存优化，用多头潜在注意力机制MLA+键值缓存优化，减少延迟。效率优化：FP8混合精度、DualPipe，减少训练时间和通信开销。
- 三阶段训练：专家孵化、专精强化、协同优化

<!-- TOC --><a name="moe-2"></a>
## MoE核心原理
https://www.youtube.com/watch?v=sOPDGQjFcuM

- MoE=专家（FFN前馈神经网络矩阵层）+路由/门控网络（开关决定token选什么专家）
- Decoder-only Transformer架构：把FFN层替换成多个专家FFN，把dense model的稠密信息进行分割，切割成很多小组专家学习模型中的信息，分组后变成sparse model（每一层只有Top k个专家被使用），经过路由选择形成path得到最终答案。
- 路由怎么选择专家：路由=FFN+softmax（计算出各专家概率去选择专家）
- 稀疏架构：Transformer分为Dense和MoE，MoE下又分Dense MoE（选所有专家）和Sparse MoE（选Top k专家后aggregate）
- 负载均衡：如果某专家计算特别快，其他专家慢，路由pathway就会自动选择快的专家，导致贫富差距加大
  - 需要KeepTopK专家选择来注入高斯噪声（有选择性地抑制某个专家过于频繁被选择，降低得分）。
  - Auxiliary Loss辅助损失（不是网络模型的损失，是负载均衡损失）：加入importance因子计算每个专家对网络的重要性，用coefficient variation去抑制最被经常使用的专家，让各专家负载均衡。
  - 专家容量：限制每个专家处理的最大token数，让网络均衡

模型规模与计算效率的tradeoff
- 2024：大参数少专家，容易训练，计算成本高，负载不均、专家利用率低。
- 2025趋势：小参数多专家（DeepSeek-V3: 256个），细粒度专家划分+动态路由优化负载均衡，计算效率高，泛化能力和扩展性强，成本低，更高参数量提升模型容量，更低推理成本（仅激活必要参数），更强任务适配性（Expert-as-a-Service）。


<!-- TOC --><a name="2-deepseek"></a>
# 2. DeepSeek

https://www.bilibili.com/video/BV1DJwRevE6d/

<!-- TOC --><a name="deepseek-v3"></a>
## DeepSeek-V3
- Multi-Head Latent Attention (MLA)：引入潜在空间提高计算效率，并保持模型对输入数据复杂关系的捕捉能力。
- Mixture of Expert (MoE)：高效专家分配（负载均衡）和计算资源利用来降低成本。
- FP8量化（混合精度训练）+多token预测（并行推理，就像随机采样，加速解码过程），提高理解能力。
- 分阶段训练，性能提升依赖于算法升级（post-training, RL，知识蒸馏）
- 通信优化DulePipe双流水线并行优化


<!-- TOC --><a name="deepseek-r1"></a>
## DeepSeek-R1
- R1-Zero（探索RL+LLM）=基于规则的奖励（准确率奖励+思考过程格式奖励）+推理为中心的大规模强化学习（组相对策略优化GRPO+瞄准Reasoning推理任务）。
- R1（工程和数据调优）=V3+GRPO，好：自我进化，具有test-time reasoning。 坏：reasoning可读性差，中英混杂。
- 相比DeepSeek-v3-Base，增强了推理链可读性（用高质量数据冷启动让RL更稳定+推理为中心的RL）、提升通用能力和安全性（拒绝采样和全领域SFT+全领域All Scenario RL）。
- 无需监督微调（节省标注成本）的纯强化学习驱动（需要强大V3基座模型+GRPO强化学习优化+推理问题可以自动化标记验证）。基于强化学习的后训练Post-Training Scaling Law，有强大推理能力和长文本思考能力。 因为大模型预训练的边际收益递减，自回归的数学推理难以自我纠错。
- 涌现出检查、反思、长链推理。即使是稀疏奖励信号，模型也能自然探索出验证回溯总结反思。
- GRPO：构建多个模型输出的群组（多个回答），计算群组内相对奖励来估计基线。相比PPO的价值函数是大模型大计算，GRPO省略了Value Model而用群组相对方式计算优势值，将策略模型与参考模型的KL散度作为正则项加入损失函数（而非奖励函数），大幅降低RL计算成本。
- 四阶段交替训练：SFT、RL、再SFT、再RL，解决冷启动和收敛效率问题。涌现出检查、反思、长链推理。
- 冷启动数据+SFT给V3，由GRPO强化后得到R1，使用rejection sampling得到reasoning data再去微调V3，几轮post-training迭代后得到R1，蒸馏出小模型。

<!-- TOC --><a name="deepseek-janus"></a>
## DeepSeek Janus
- Janus：把理解和生成任务合并统一成自回归Transformer架构。understanding encoder（用SigLIP对图像编码）和generation encoder（用VQ-tokenizer）不同，二者编码后都经过adaptor进入自回归LLM架构，因此提升模型灵活性同时缓解生成-理解的冲突。
- Janus-Flow：不同于Janus，Generation部分基于Rectified Flow（如Stable Diffusion3）在LLM内融合两种架构，encoder-decoder迭代配对，统一视觉理解与文本生成。
- Janus-Pro：用ImageNet充分训练，用多模态数据后训练。


<!-- TOC --><a name="kimi-k15"></a>
## Kimi-K1.5
- 强化学习让模型试错。In-Context RL不训练模型规划，而是模拟规划approximate planning（将每个state和value都视为language tokens，建模成contextual bandit问题，用REINFORCE变种来优化，并用长度惩罚机制防止overthinking算力损耗）。
- 采样策略：课程学习循序渐进+优先采样做难题。
- 四阶段：Pretraining、SFT、Long CoT SFT、RL。
- 构造Vision Data（把文本内容转化为视觉格式）+Long2Short蒸馏（用够短思维链达到长思维链效果：模型融合+最短拒绝采样+DPO）。

技术对比
- Kimi K1.5与DeepSeek-R1对比：Kimi K1.5更多从In-Context RL出发，直接训练模型approximate planning，将思考过程建模到语言模型的next token prediction中，但复杂推理可能难以迭代。DeepSeek-R1从纯RL出发，用GPRO+Rule-based reward激活模型本身的推理能力。二者都根据答案对错给予奖励惩罚。
- 蒸馏与强化学习对比：蒸馏就是学生学习强大老师，短期内掌握复杂技能。强化学习是试错尝试，泛化性更好，因为SFT主要负责记忆而非理解故很难分布外泛化。DeepSeek用R1蒸馏出的小模型，因为R1强大，发现了高阶推理范式（小模型直接RL则难以发现因为预训练知识不足），蒸馏甚至超过RL的方法。

未来展望
- 长思维链的欺骗性推理In-Context Scheming、奖励篡改。要引入AI-driven监督机制、对比推理。
- 模态扩展+模态穿透（强推理+多模态），拓展模型推理边界。因为RLHF+DPO模态无感，用从语言反馈中学习Learning from Language Feedback，捕捉人类意图的多元偏好和复杂模态交互，实现any-to-any models与人类意图对齐。
- 强推理赋能Agentic，要反思、长程规划、工具调用。
- 大模型就像压缩器，因为弹性而抗拒对齐，用Deliberative Alignment审计对齐把宪法融入到模型推理过程。
- LLM受限于过程性推理任务，人类可以抽象出高维概念并细粒度反馈。

<!-- TOC --><a name="3-deepseek-open-source-week-20250224"></a>
# 3. DeepSeek Open Source Week, 2025.02.24

> https://github.com/deepseek-ai/open-infra-index

> https://zhuanlan.zhihu.com/p/27181462601

- FP8计算生态崛起，推动大模型算法向低精度迁移
- MoE模型加速落地，未来scaling law全面转向MoE架构，聚焦AI Infra底层

<!-- TOC --><a name="day-1-flashmla"></a>
## Day 1. FlashMLA
专为Hopper架构优化的MLA Kernel，支持变长序列
- Multi-head Latent Attention (MLA): boost inference efficiency，比Multi-head Attention (MHA)多了latent隐变量，通过潜在向量latent vector压缩了Key-Value cache，大大降低训练耗时和显存、提高吞吐。
- MLA架构：用low-rank key-value joint compression加速，类似LoRA低秩分解，把KV Cache大参数压缩成小latent向量来节省KV显存，再反压缩回来。
- FlashMLA：分页KV缓存管理、异步内存拷贝（英伟达Hopper架构）、双模式执行引擎（动态负载均衡，对长短序列用不同缓存模式）
- 应用：MoE推理加速（长文本生成，对话系统）
- 意义：降低高性能推理门槛，低成本部署
- 技术：融合FA 2/3注意力优化与cutlass硬件适配，针对Hopper Tensor Core定制

<!-- TOC --><a name="day-2-deepep"></a>
## Day 2. DeepEP
MoE专用通信库，支持EP、FP8低精度通信，实现高吞吐低延迟GPU Kernel
- All-to-all通信，让输入可以正确分发给各专家，并将专家输出结果正确聚合。MoE架构正适合All-to-all (dispatch分发 + combine聚合)。
- 应用：MoE训练与分布式推理
- 意义：突破MoE专家间通信瓶颈，推动千亿级模型实用化部署，AI行业将从堆算力转为智能优化
- 技术：基于NVSHMEM优化通信协议，结合自研低精度专家分发算法

<!-- TOC --><a name="day-3-deepgemm"></a>
## Day 3. DeepGEMM
高性能FP8矩阵运算库，Hopper峰值性能1350+tflops，显存占用较FP16减半
- 应用：MoE架构计算优化，边缘设备轻量化部署
- 意义：推动行业向低精度计算迁移，解决千亿模型内存墙问题
- 技术：集成Hopper架构FP8 Tensor Core指令集，动态量化策略平衡精度与效率

<!-- TOC --><a name="day-4-dualpipe"></a>
## Day 4. DualPipe
双向流水线并行框架，通过计算与通信重叠减少流水线气泡，提升训练效率
- 应用：超大规模MoE训练（DeepSeek V3 128k上下文）
- 意义：解决流水线并行中资源闲置问题，降低训练成本
- 技术：创新流水线调度算法，结合跨节点全通信优化

<!-- TOC --><a name="day-5-3fs"></a>
## Day 5. 3FS
分布式训练存储优化方案，支持高效数据分片与缓存管理
- 应用：超大规模训练数据加载加速、分布式检查点存储
- 意义：缓解训练I/O瓶颈，提升GPU集群利用率
- 技术：推测采用分阶段数据预加载与内存映射技术


<!-- TOC --><a name="acknowledgements"></a>
# Acknowledgements

> https://space.bilibili.com/517221395/upload/video

> https://github.com/chenzomi12/AIFoundation/
