Generative AI on AWS, by Chris Fregly, 2024

<img src="https://github.com/user-attachments/assets/c04f9aa4-6563-4e9b-ad78-03ebd36cc9ec" width="32%" height="32%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [4. 显存和计算优化](#4-)
- [7. RLHF](#7-rlhf)
- [10. 多模态](#10-)
- [11. Stable Diffusion受控生成和微调](#11-stable-diffusion)

<!-- TOC end -->

<!-- TOC --><a name="4-"></a>
## 4. 显存和计算优化

优化自注意力层
- FlashAttention：把自注意力性能提高2-4倍，时间复杂度O(n)，减少80%-90%显存占用，Transformer可以处理更长输入序列，
- GQA分组查询注意力：将Q分组到更少的KV头中，减少注意力头的显存占用。

分布式GPU集群
- 分布式数据并行Distributed Data Parallel（DDP）：PyTorch内置DDP实现可将模型自动复制到每个GPU，将数据切分成批次，批次并行发到每个GPU。
- 全分片数据并行Fully Sharded Data Parallel（FSDP）：论文ZeRO (Zero Redundancy Optimizer零冗余优化器)，通过GPU之间分片模型及其额外梯度、激活值、优化器状态，减少DDP数据冗余，实现系统零冗余。三阶段：
  - GPU之间分片优化器状态，但仍可将模型显存占用减少为1/4。
  - GPU之间分片优化器状态和梯度，将GPU显存减少为1/8。
  - GPU之间分片所有内容（包括模型参数），GPU显存减少为1/n，n = GPU数量。
  - 全分片提供最佳显存节省，代价是增加GPU通信开销，将分片系数设为两者（与全量复制未分片）之间的任何值都将启用混合分片。
 
<!-- TOC --><a name="7-rlhf"></a>
## 7. RLHF

- 奖励模型RoBERTa：检测有害Hate语言，预测文本的讨厌、不讨厌概率分布。可以用来微调LLM并减少有害性。
- 近端策略优化PPO：每次迭代PPO都对模型权重小幅有界更新，让微调更稳定，最终很好地泛化新输入。
- PPO + RLHF：把提示传给模型生成补全，将提示-补全对传给奖励模型，奖励模型提供一组logit和在不讨厌、讨厌类别上的概率分布，希望优化不讨厌类别。RLHF每次迭代都更新模型权重，一段时间后LLM获得更高奖励，因为生成有害补全更少，迭代一直持续到模型与人类价值对齐（有害性评分阈值）或到最大迭代次数。
- 缓解奖励破解：KL散度量化RLHF后的补全与（冻结复制模型权重并当做不可变的参考模型）补全之间的距离，若微调模型破坏奖励并生成与序列相差太大的token（无意义但低害），则RL用低奖励惩罚微调模型。
- PEFT + RLHF可减少PPO计算
- 评估：`aggregate_toxicity_scores()`计算有害性评分均值标准差，比较RLHF前后分数。
- 蒸馏损失：老师预测token是soft label，学生预测token是soft prediction；学生预测结果是hard prediction，数据真实值是hard label，二者差异是student loss；将蒸馏损失+学生损失，更新学生权重。
- 最大边际相关性Max Marginal Relevance (MMR)：内置于向量数据库的重排序算法，保持检索结果与输入prompt相关的同时减少检索结果冗余信息（因为检索结果经常很相似），提高差异化的上下文信息。

<!-- TOC --><a name="10-"></a>
## 10. 多模态

- CLIP (Contrastive Language-Image Pre-training)定量评估：评分图像与语义的相似性相关性、Frechet Inception距离（FID：两个图像数据集相似性）
- 扩散模型：U-Net（前向扩散加噪 + 反向扩散降噪）
- Stable Diffusion：是Latent Diffusion Model
  - 文本编码器：OpenCLIP，输入文本转换为token特征
  - 扩散过程：在隐空间计算，快，输入随机隐空间种子（选择种子控制噪声随机分布）、U-Net采样步数（逐步采样最终对齐图像与文本特征）
  - 文本条件控制：在网络层间添加注意力层，处理输入给扩散模型的文本
  - 交叉注意力：视觉颜色形状的KV + 语言token输入的Q，把文本和图像特征融合
  - 采样器：反复对输入的随机隐空间种子图像去噪，控制去噪步数。训练的不同时间，采样器根据加噪参数和样本更新规则来加噪，每次迭代调整噪声并逐步降低。
  - 图像解码器：自编码模型，把去噪后的隐空间表征转化为图像输出

<!-- TOC --><a name="11-stable-diffusion"></a>
## 11. Stable Diffusion受控生成和微调

- ControlNet：输入学习任务（边缘检测、深度映射），只要少量数据即可训练强大控制器，让Stable Diffusion输出图像遵循控制引导（如海滩景色、逼真高质、彩色照片、城市夜晚等风格）。
- DreamBooth微调：仅用3-5个样本定制个性化文生图模型。结合类别特定的先验保留损失，可并行微调文生图扩散模型。适用于营销语境化、画家风格创意、加装饰。
  - 结合LoRA微调：可以微调扩散模型的所有参数，保持文本转换器冻结，产生新扩散模型
  - 轻量级微调：文本反演Textual Inversion。通过向量空间学习表征新概念，保持原有预训练文本-图像模型冻结，仅需few-shot风格示例图像，即可个性化定制文生图，无需改动基础模型。
- RLHF：Denoising Diffusion Policy Optimization (DDPO)，强化学习中每个去噪步骤都是一个行动，DDPO关注整个去噪步骤序列，更好地最大化最终生成图像的奖励。

