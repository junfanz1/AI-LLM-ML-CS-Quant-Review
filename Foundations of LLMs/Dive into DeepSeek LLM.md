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

缓解梯度消失（无法捕捉长距信息）梯度爆炸（参数更新不稳定不收敛）
- LSTM记忆单元，保留长距信息
- 梯度裁剪Gradient Clipping：若反向传播计算梯度的范数超过阈值，则缩放梯度
- Xavier/He初始化权重，初始权重方差在网络层之间保持平衡
- 残差连接：允许梯度直接传播，减少梯度衰减（ResNet）
- Batch Normalization, Layer Normalization稳定激活值分布，缓解梯度爆炸。正则化方法（Dropout）防止过拟合，减少极端值，增加泛化能力
- 调整学习率、优化器选择（Adam自动调整学习率，控制梯度波动）

Transformer Encoder
- 多头自注意力：从不同子空间学习多维度语义关联
- 前馈神经网络：独立作用于每个位置，用GELU增加表达能力，进一步提取深层语义特征
- 残差连接、层归一化：缓解梯度消失，促进模型收敛；稳定训练过程，加速模型优化

Transformer Decoder
- Masked Multi-Headed Self-Attention：生成过程的自回归性，防止模型访问未来词信息
- Encoder-Decoder Attention：有效融合源语言信息，用编码器输出作为KV，解码器隐藏状态作为Q，实现跨序列以来建模
- 前馈神经网络

<!-- TOC --><a name="4-rl-deepseek-r1-zero"></a>
## 4. RL & DeepSeek-R1-Zero

强化学习
- PPO (Proximal Policy Optimization)：最大化策略的语气汇报，策略更新的目标是控制策略更新幅度（裁剪策略）避免不稳定
  - 分布式强化学习：可以用`DistributedRLTrainer`类，管理分布式训练过程（计算优势函数、回报、更新策略），每个线程在环境中训练，并通过共享经验加速全局模型训练。
  - 训练时用Python threading多线程来模拟多智能体并行训练，每个智能体在自己环境训练，最终合并全局奖励。
```py
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.distributions import Categorical 
import gym 
from collections import deque 

# MLP, to generate each timestep action distribution, output softmax prob distribution for sampling
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.softmax(x, dim=-1)

# compute returns, compute advantages, update with clippinng strategy to restrict each update size
class PPO:
    def __init__(self, env, policy_network, gamma = 0.99, lr=3e-4, epsilon=0.2, batch_size=64, epochs=10):
        self.env = env 
        self.policy_network = policy_network
        self.gamma = gamma 
        self.lr = lr 
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.epochs = epochs 
        self.optimizer = optim.Adam(policy_network.parameters(), lr=lr)

    def compute_returns(self, rewards, dones, values, next_value):
        returns = []
        R = next_value
        for r, done in zip(rewards[::-1], dones[::-1]):
            R = r + self.gamma * R * (1-done)
            returns.insert(0, R)
        return torch.tensor(returns)
    
    def compute_advantages(self, rewards, dones, values, next_value):
        returns = self.compute_returns(rewards, dones, values, next_value)
        advantages = returns - values 
        return advantages
    
    def update(self, states, actions, log_probs_old, returns, advantages):
        for _ in range(self.epochs):
            log_probs = self.policy_network(states).gather(1, actions.unsqueeze(-1))
            ratios = torch.exp(log_probs - log_probs_old)
            surrogate = ratios * advantages
            clipped_surrogate = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            loss = -torch.min(surrogate, clipped_surrogate).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def train(self, num_episodes):
        all_rewards = []
        for episode in range(num_episodes):
            states = []
            actions = []
            rewards = []
            log_probs = []
            dones = []
            values = []
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False 
            total_reward = 0 

            while not done:
                state = state.unsqueeze(0)
                dist = self.policy_network(state)
                m = Categorical(dist)
                action = m.sample()
                log_prob = m.log_prob(action)
                value = dist[0, action]

                next_state, reward, done, _ = self.env.step(action.item())
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                log_probs.append(log_prob)
                dones.append(done)
                values.append(value)

                state = torch.tensor(next_state, dtype=torch.float32)
                total_reward += reward 

            all_rewards.append(total_reward)
            returns = self.compute_returns(rewards, dones, values, value)
            advantages = self.compute_advantages(rewards, dones, values, value)
            self.update(torch.stack(states), torch.tensor(actions), torch.stack(log_probs), returns, advantages)

            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
            return all_rewards
```

- TRPO (Trust Region Policy Optimization)：每次更新时限制策略变化，约束KL散度来保证策略处在信任范围内，避免参数大幅变化。
- DPO (Differentially Private Optimization)：结合隐私保护和策略优化，在策略优化中引入差分隐私，通过加噪声（可能带来性能损失）或调整梯度来确保agent学习中不泄露太多用户信息，防止敏感数据泄露。
- GRPO (Generalized Reinforcement Policy Optimization)：用灵活的优化框架，允许agent根据任务特定调整策略，能处理复杂约束条件（多目标优化），适合复杂任务。

数据采样：Greedy Sampling（选当前最优动作）、epsilon-Greedy Sampling（大部分选最优，偶尔选随机探索）、Boltzmann采样（根据Q值计算概率，agent在各动作间进行概率选择）

RL优化策略，提高收敛和稳定性
- Experience Replay经验回放：存储历史经验，打破数据相关性，提高数据利用率，加速收敛。（代码见P121）
- 目标网络Target Network：缓解Q值过度估计，每次更新不直接依赖当前策略结果，而是依赖目标网络输出
- Entropy Regularization策略熵正则化：保持随机探索，避免太早收敛，适合基于梯度的方法（A3C、PPO）
- n-step Learning：多步汇报来估计值函数，加速收敛
- Adaptive Learning Rate

DeepSeek-R1-Zero奖励模型：基于深度学习的动态奖励生成，通过对环境和agent行为深度建模，自适应地为每个决策过程设计合适的奖励。
- `RewardModel`类：简单神经网络，预测给定状态-动作对的奖励信号（代码见P106）
- `RewardBasedLearning`类：基于奖励建模的强化学习，计算每个动作返回值和优势函数，结合策略更新和奖励模型优化，改进策略并优化奖励机制。（代码见P106）
- `AdaptiveRewardModel`类：自适应奖励模型，根据当前状态-动作对生成奖励信号，提供agent训练实时反馈。（代码见P110）
- `AdaptiveRewardLearning`类：结合PPO策略优化和奖励建模的更新机制，训练agent策略网络，同时优化奖励生成模型。（代码见P110）

奖励策略稀疏性的改进
- Reward Shaping奖励塑形：设计潜在奖励函数引导agent行为，提供中间奖励。
  - `RewardShapingModel`类：动态调整奖励信号，生成更适应当前学习阶段的奖励。（代码见P115）
  - `RewardLearning`类：结合PPO策略优化与奖励塑形，对策略网络和奖励模型共同训练，优化agennt策略和奖励信号。（代码见P115）
- 调整探索策略，如Entropy Regularization熵正则化，保持策略随机性，让agent全面探索
- 基于模型的增强学习：构建环境模型，让agent在模拟环境中预测，获得更多训练样本和奖励信号，即便奖励信号稀疏也可以通过模拟环境来加速学习
- n-step Learning：聚合未来多步的奖励信号
- 多任务并行训练：`MultiTaskModel`，多个相关任务通过共享模型参数加速学习（代码见P121）。用预训练BERT作为共享编码器，为数学推理和代码生成分别构建专用全连接层，输出层映射到词汇表大小用于模拟文本生成。（代码见136）


<!-- TOC --><a name="5-cold-start-rl-deepseek-r1"></a>
## 5. Cold-Start RL & DeepSeek-R1

DeepSeek-R1冷启动策略
- 自监督预训练迁移学习
  - `TransferLearningModel`类：全连接神经网络，前两层是加载预训练权重，最后一层是任务专用的分类层。（代码P130）
  - `TransferLearningTrainer`类：迁移学习训练流程，包括预训练权重加载、模型微调、训练、验证。（代码P131）
- 离线强化学习
- 元学习快速适应（Model-Agnostic Meta-Learning：多个小样本快速迭代，学会高效参数调整，只要少量梯度更新即可快速适应模型初始化、策略优化）
- 知识蒸馏
- KV数据缓存与复用

拒绝采样与监督微调
- 拒绝采样：生成多个结果后基于一定评分准则筛选最优解，避免模型输出重复无意义内容。
- 损失函数 = 监督学习用交叉熵损失 + 强化学习用策略梯度计算生成样本与目标奖励之间的偏差并反向传播。用预训练知识任务适配、训练中自适应探索更优输出策略。

【监督学习+强化学习的微调】用DeepSeek-R1模型参数模拟加载，用合成数据监督微调，结合奖励信号进行策略优化，用Flask提供推理API服务。

```py
import os 
import time 
import random 
import json 
import numpy as np 
from typing import Tuple, List 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader, ConcatDataset 
from torch.distributions import Categorical 
from transformers import BertModel, BertTokenizer 
from flask import Flask, request, jsonify

# 1. Simulate DeepSeek-R1 API parameter loading and pre-training weights obtain
def simulate_deepseek_r1_pretrained_weights() -> dict:
    """
    Simulate DeepSeek-R1 pretraining weights, return dictionary
    Pretraining weight are reused to initialize model's shared encoder
    """
    weights = {
        "encoder.fc.weight": torch.randn(768, 768),
        "encoder.fc.bias": torch.randn(768),
    }
    print("Simulate obtaining DeepSeek-R1 pretraining weights, success!")
    return weights 

# 2. Dataset build: create Q&A mission dataset 
class QA_Dataset(Dataset):
    """
    Synthesize Q&A dataset, each sample containing Q&A texts.
    For SFT, small data, good for cold start.
    """
    def __init__(self, num_samples: int=300):
        self.num_samples = num_samples
        self.samples = []
        # generate easy Q&A
        for _ in range(num_samples):
            a = random.randint(1, 50)
            b = random.randint(1, 50)
            question = f"Calculate {a} + {b} = ?"
            answer = str(a + b)
            self.samples.append((question, answer))
        # non-math Q&A
        extra_samples = [
            ("Is Beijing the capital of China?", "Yes, Beijing is the capital of China."),
            ("What's water's chemical representation?", "H2O"),
            ("Where does the sun rise?", "East"),
            ("Is the moon the satellite of earth?", "Yes.")
        ]
        self.samples.extend(extra_samples)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]
    
# 3. Multi-task model definition: SFT + RL combined training 
class FineTuneRLModel(nn.Module):
    """
    Model Architecture:
    - Use pretrained BERT as shared encoder (simulate DeepSeek-R1)
    - Task head: split to SFT task head and policy reward task head 
    - Output layer will map task head outputs to vocabulary table size (simulate text generation)
    """
    def __init__(self, hidden_size: int=768, vocab_size: int=30522):
        super(FineTuneRLModel, self).__init__()
        # use pretrained BERT as encoder
        self.encoder = BertModel.from_pretrained("bert-base-chinese")
        # SFT head: for supervised learning part, share encoder outputs then fullly-connected layer 
        self.supervised_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # policy head: for RL part, similar structure
        self.rl_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        # output layer, map task head output to vocabulary table
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask, mode="supervised"):
        """
        Forward propagation, select different task head according to mode 
        mode="supervised": use supervised tuning task head 
        mode="rl": use policy reward task head
        """
        # use shared encoder to get text representation 
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_representation = encoder_outputs.last_hidden_state[:, 0, :] # [CLS] vector 
        if mode == "supervised":
            head_output = self.supervised_head(cls_representation)
        elif mode == "rl":
            head_output = self.rl_head(cls_representation)
        else:
            raise ValueError("mode must be 'supervised' or 'rl'")
        logits = self.output_layer(head_output)
        return logits 
    
# 4. data preprocessing and Collate function 
def collate_fn_qa(batch: List[Tuple[str, str]], tokenizer, max_length: int=32) -> dict:
    """
    Collate function, process Q&A dataset 
    encoder question text to input_ids, attention_mask, encoder answer text to labels
    """
    questions = [q for q, a in batch]
    answers = [a for q, a in batch]
    enc_inputs = tokenizer(questions, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    enc_labels = tokenizer(answers, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    return {
        "input_ids": enc_inputs["input_ids"],
        "attention_mask": enc_inputs["attention_mask"],
        "labels": enc_labels["input_ids"]
    }

# 5. train combined functions: supervised loss + policy gradient loss 
def compute_rl_loss(logits, labels, rewards, tokenizer):
    """
    RL loss:
    - logits: model output logits 
    - labels: true label token 
    - rewards: rewards for generated text (simulated as random or fixed reward)
    use weighted cross entropy loss for policy gradient 
    """
    criterion = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id, reduction="none")
    ce_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    # simulate reward adjustments: multiply by reward factor
    # assume rewards as a scalar, apply to all samples
    rl_loss = (ce_loss * rewards).mean()
    return rl_loss 

def train_finetune_rl(model, tokenizer, dataloader, device, epochs: int=3, rl_weight: float=0.5):
    """
    Train function, use SFT & RL to update, one by one 
    each batch, calcualte supervised loss and rl loss, weighted sum, then back propagate
    """
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    total_steps = epochs * len(dataloader)
    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} / {epochs}")
        for batch in dataloader:
            # obtain data batch, containing input_ids, attention_mask, labels
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # forward propagation, supervised part 
            logits_sup = model(input_ids, attention_mask, mode="supervised")
            # supervised cross entropy loss 
            loss_sup = nn.CrossEntropyLoss(ignore_index = tokenizer.pad_token_id)(logits_sup.view(-1, logits_sup.size(-1)), labels.view(-1))
            # forward propagation RL part (can use same input, or design special input)
            logits_rl = model(input_ids, attention_mask, mode="rl")
            # simulate reward signal: randomly generate reward coefficient in [0.8, 1.2]
            rewards = torch.tensor(random.uniform(0.8, 1.2), device=device)
            loss_rl = compute_rl_loss(logits_rl, labels, rewards, tokenizer)

            # total loss = supervised loss + rl loss, weighted sum
            total_loss = loss_sup + rl_weight * loss_rl
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            step += 1 
            if step % 10 == 0:
                print(f"Step {step} / {total_steps}, Supervised Loss: {loss_sup.item():.4f}, RL Loss: {loss_rl.item():.4f}, Total Loss: {total_loss.item():.4f}")
    print("Combined fine tuning done.")

# 6. Reasoning function and API: combined SFT & rejection sampling 
def generate_text(model, tokenizer, input_text: str, mode: str="supervised", max_length: int=32) -> str:
    """
    Generate response based on input text, suppoert supervised or RL modes
    """
    model.eval()
    device = next(model.parameters()).device 
    encodings = tokenizer(input_text, return_tensors="pt", truncation=True, max_length = max_length)
    input_ids = encodings["input_ids"].to(device)
    attention_mask = encodings["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, attention_mask, mode=mode)
    # rejection sampling to control output (use top-k as simple simulation)
    top_k = 50 
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k, dim=-1)
    # randomly choose token 
    chosen_idx = top_indices[0, random.randint(0, top_k-1)].item()
    generated_text = tokenizer.decode([chosen_idx], skip_special_tokens=True)
    return generated_text

# 7. Flask API service deployment 
app = Flask(__name__)
# gloabl variable: load tokenizer and model after fine-tuning 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Current device: {DEVICE}")
TOKENIZER = BertTokenizer.from_pretrained("bert-base-chinese")
MODEL = FineTuneRLModel()
MODEL.to(DEVICE)

# simulate load pretrained weights (call simulate function)
pretrained_weights = simulate_deepseek_r1_pretrained_weights()
if pretrained_weights is not None:
    model_dict = MODEL.state_dict()
    # only load encoder part weights 
    for key in pretrained_weights:
        if key in model_dict:
            model_dict[key] = pretrained_weights[key]
    MODEL.load_state_dict(model_dict)
    print("pretrained weight reloaded to FineTuneRLModel, done.")
else:
    print(" pretrained weight not loaded")

@app.route('/finetune_inference', methods=['POST'])
def finetune_inference():
    """
    Inference API: take in JSON request {"text": "input text", "mode": "supervised" or "rl"}
    return generated response text
    """
    data = request.get_json(force=True)
    input_text = data.get("text", "")
    mode = data.get("mode", "supervised")
    if not input_text:
        return jsonify({"error": "Lack 'text' parameter"}), 400 
    output_text = generate_text(MODEL, TOKENIZER, input_text, mode=mode)
    return jsonify({"reply": output_text})

@app.route('/', methods=['GET'])
def index():
    return "Fine-tune RL inference service start, use /finetune_inference API for inference"

# 8. Main function and training pipeline calls 
def main():
    # create QA data for SFT 
    qa_dataset = QA_Dataset(num_samples=300)
    # use collate_fn_qa to encode data
    qa_loader = DataLoader(qa_dataset, batch_size=8, shuffle=True, collate_fn=lambda batch: collate_fn_qa(batch, TOKENIZER, max_length=32))
    # combined training, SFT cross entropy loss + RL loss for fine tuning 
    print("start combining fine tuning for supervised learning and RL")
    train_finetune_rl(MODEL, TOKENIZER, qa_loader, DEVICE, epochs=3, rl_weight=0.5)

    # start Flask API service for online inference
    print("start fine tuning inference service...")
    app.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
```

全场景强化学习
- 对抗训练：引入扰动数据，增强模型对不确定输入的鲁棒性，减少特定模式的过拟合，提高未见数据适应力。
- 迁移学习：多领域数据预训练，结合少量目标领域数据微调，让模型更快适应新任务，减少冷启动数据需求
- 自适应：对金融、医疗、计算机领域，DeepSeek-R1在训练中加入任务相关自适应权重，在特定领域表现更优。增量学习，权重动态更新。
- Hierarchical Agent分层强化学习：agent首先调用高层策略（选子目标，从预定的集合中选择），然后让低层策略（根据当前状态与子目标选择具体动作）执行动作直到达到子目标或超出步数限制，二者都可用策略梯度更新，最后根据奖励信号更新高层与低层策略网络。（代码P152）

【知识蒸馏：训练采用温度调节temperature scaling、损失加权、梯度剪裁、学习率调度。两阶段：先基础蒸馏，用KL散度损失和交叉熵损失联合优化；然后对特定任务微调，提高学生在下游任务表现。教师模型输出软目标指导学生，在测试集评估学生性能，实现准确性+推理速度双优化】

```py
import os 
import time 
import random 
import numpy as np 
from typing import List, Tuple 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torch.optim.lr_scheduler import StepLR 
from torch.utils.data import Dataset, DataLoader 
from flask import Flask, request, jsonify

# 1. create dataset: use synthetic classification dataset
class SynthetheticClassificationDataset(Daset):
    """
    Each sample is randomly generated feature embedding and corresponding label
    To simulate downstream classification task, good for distillation training
    """
    def __init__(self, num_samples: int=1000, input_dim: int=100, num_classes: int=10):
        self.num_samples = num_samples 
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.data = np.random.randn(num_samples, input_dim).astype(np.float32)
        # randomly generate 0 ~ num_classes-1 integers as labels 
        self.labels = np.random.randint(0, num_classes, size=(num_classes,)).astype(np.int64)

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def collate_fn(batch: List[Tuple[np.ndarray, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Collate function: turn data in batch to Tensor
        """
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        return features, labels 
    
# 2. Define Teacher and Student 
class TeacherModel(nn.Module):
    """
    Teacher is large, generate soft label
    model simulates DeepSeek-R1 some abilities
    """
    def __init__(self, input_dim: int=100, hidden_dim: int=512, num_classes: int=10):
        super(TeacherModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits 
    
class StudentModel(nn.Module):
    """
    Student is small, distilled from teacher, small parameters but match the teacher's performance
    """
    def __init__(self, input_dim: int=100, hidden_dim: int=128, num_classes: int=10):
        super(StudentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits
    
# 3. Define knowledge distillation training function
def train_distillation(teacher: nn.Module, student: nn.Module, dataloader: DataLoader, device: str="cpu", epochs: int=5, temperature: float=2.0, alpha: float=0.7) -> None:
    """
    temperature: to soft prob distribution of teacher output 
    alpha: distillation loss weight, a weighted factor w.r.t. hard label loss
    """
    teacher.to(device)
    student.to(device)
    teacher.eval() # teacher model is fixed, no update

    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    # loss function
    ce_loss_fn = nn.CrossEntropyLoss()
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")

    total_steps = epochs * len(dataloader)
    step = 0

    for epoch in range(epochs):
        running_loss = 0.0 
        for batch in dataloader:
            inputs, hard_labels = batch 
            inputs = inputs.to(device)
            hard_labels = hard_labels.to(device)

            # teacher model generate soft label for temperature scaling 
            with torch.no_grad():
                teacher_logits = teacher(inputs) / temperature
                teacher_soft = F.softmax(teacher_logits, dim=-1)
            
            # student model forward propagation (two modes both use temperature to adjust output)
            student_logits = student(inputs) / temperature

            # distillation loss: KL divergence loss, note that input is log prob
            distill_loss = kl_loss_fn(F.log_softmax(student_logits, dim=-1), teacher_soft)
            # hard label loss: cross entropy loss (no temperature scale)
            hard_loss = ce_loss_fn(student(inputs), hard_labels)
            # total loss is weighted sum of two losses, multiplied by temperature squared 
            total_loss = alpha * (temperature ** 2) * distill_loss + (1 - alpha) * hard_loss 

            optimizer.zero_grad()
            total_loss.backward()
            # gradient pruning to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += total_loss.item()
            step += 1 
            if step % 10 == 0:
                avg_loss = running_loss / 10 
                print(f"Epoch [{epoch + 1}/{epoch}], Step [{step}/{total_steps}], Loss: {avg_loss:.4f}")
                running_loss = 0.0 
        scheduler.step()
    print("Distillation training complete.")

def evaluate_model(mode: nn.Module, dataloader: DataLoader, device: str="cpu") -> float:
    # evaluate model accuracy on testing set 
    model.to(device)
    model.eval()
    total_correct = 0 
    total_samples = 0 
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch 
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
    accuracy = total_correct / total_samples
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    return accuracy 

# 5. Flask API Deployment ...
```


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
