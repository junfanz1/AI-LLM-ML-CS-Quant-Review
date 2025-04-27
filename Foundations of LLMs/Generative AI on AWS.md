Generative AI on AWS, by Chris Fregly, 2024

<img src="https://github.com/user-attachments/assets/c04f9aa4-6563-4e9b-ad78-03ebd36cc9ec" width="32%" height="32%">

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
 
## 7. RLHF

- 奖励模型RoBERTa：检测有害Hate语言，预测文本的讨厌、不讨厌概率分布。可以用来微调LLM并减少有害性。
- 近端策略优化PPO：

