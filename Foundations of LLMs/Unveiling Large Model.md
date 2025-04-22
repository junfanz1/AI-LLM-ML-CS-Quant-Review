Unveiling Large Model, by Liang Wen, 2025

<img src="https://github.com/user-attachments/assets/65fd0bf1-dde7-4282-a7aa-fd413812c686" width="34%" height="34%">

（不全面的笔记）

## 1. 压缩即智能

GPT训练过程本质是对整个数据集的无损压缩。规模越大越智能，因为大模型的低损失实现更高压缩率。

## 5. Llama

- Pre-normalization：用RMSNorm作为归一化函数，每个Transformer子层之前对输入归一化
- SwiGLU：SWISH+GLU，优化Transformer前馈网络
- 旋转位置编码RoPE：提高模型外推能力，将相对位置信息集成到自注意力中的绝对位置编码方式
- AdamW优化器：将weight decay从梯度更新中分离出来，提高优化效果。用cosine learning rate decay根据余弦曲线动态调整学习率，避免学习率在切换时产生动荡。
- RLHF：通过rejection sampling和PPO迭代优化。
  - 强化学习微调预训练模型：近端策略优化Proximate Policy Optimization (PPO)，用奖励模型评估回答的分数。目标函数 = 强化学习输出不要偏离有监督微调太多 + 保证微调效果的同时让语言模型在通用能力上效果不变差。
  - Llama 2奖励函数 = 有用性 + 安全性。将人类偏好数据转换为二元排序标签格式（选择和拒绝），用binary ranking loss优化奖励模型；加入边际项，根据差异大小的回答对分配边际值，提高在区分度高的样本上的准确度。
- 预训练数据：对较高准确性的数据up-sampling，增加模型知识和减少幻觉。
- 分组查询注意力Grouped-query Attention (GQA)：提高大模型推理可扩展性。

## 8. 大模型训练优化

- 









