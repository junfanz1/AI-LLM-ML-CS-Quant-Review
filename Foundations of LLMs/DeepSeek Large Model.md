
DeepSeek Large Model High-Performance Core Technology and Multimodal Fusion Development, by Xiaohua Wang, 2025

<img src="https://github.com/user-attachments/assets/1c3c59ff-4b22-4163-a12b-cfc027950602" width="32%" height="32%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [3. Attention 1](#3-attention-1)
- [4. Attention 2：RoPE](#4-attention-2rope)
- [5. Attention 3：MoE](#5-attention-3moe)
- [6. Attention 4：MLA](#6-attention-4mla)
- [7. DeepSeek API Calling](#7-deepseek-api-calling)
- [8. DeepSeek Fine-tuning](#8-deepseek-fine-tuning)
- [9. Diffusion](#9-diffusion)
- [10. Multimodal Fusion](#10-multimodal-fusion)
- [11. Cross-Attention, Audio](#11-cross-attention-audio)
- [12. Token Compression](#12-token-compression)
- [13. Image Encoder: VQ-VAE, FSQ](#13-image-encoder-vq-vae-fsq)
- [14. torchvision Video Classification](#14-torchvision-video-classification)
- [15. DeepSeek Agents](#15-deepseek-agents)

<!-- TOC end -->

<!-- TOC --><a name="3-attention-1"></a>
## 3. Attention 1

- 注意力机制是从大量信息中选择筛选出少量重要信息，并聚焦在重要信息上，忽略不重要信息。
- Layer Normalization是对同一序列不同位置的数据进行归一化，Batch Normalization是对一个batch中不同序列中处于同一位置的数据进行归一化。
- QKV：用Query, Key之间相似度来确定注意力权重，基于缩放点积softmax对权重转换，然后用Value加权求和。注意力函数是一个Query到一系列Key-Value对的映射。
- 分组查询注意力Group Query Attention (GQA)：共享键和值矩阵，减少显存占用，提高推理速度，适合长序列大模型。
- 多头潜在注意力MLA：低秩压缩（高维矩阵->多个低维矩阵乘积），降低KV cache需求，高效推理且高质量输出。

Autoencoder = Input embedding + positional encoding + multi-head attention + layer normalization + feed forward network

```py
import torch
import torch.nn as nn 
import math

class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9) # give small number where mask==0
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn) # apply dropout to attention weights
        # weighted sum for values
        return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h 
        self.h = h 

        # 3 Linear layers for QKV 
        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model) # combine outputs into one vector 
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # multi-head outputs combine to one vector, and apply to linear layer
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)
    

class SublayerConnection(nn.Module):
    """
    Residual connection with layer normalization
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = torch.nn.LayerNorm(size) # size is input dim 
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        Apply residual connection for all same-size sublayer.
        x: input tensor 
        """
        # layer normalize x, then pass to sublayer, apply dropout, connect initial x by residual connection 
        return x + self.dropout(sublayer(self.norm(x)))
    

class PositionwiseFeedForward(nn.Module):
    """
    FFN: two-layer fully connected network
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        # initialize first fully-connected layer, input dim d_model, output dim d_ff 
        self.w_1 = nn.Linear(d_model, d_ff)
        # initialize second fully-connected layer, intput dim d_ff, output dim d_model 
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = torch.nn.GELU()

    def forward(self, x):
        """
        feed forward: input x, through first fully connected layer, activation function, dropout layer, and second fully connected layer.
        """
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    

class TransformerBlock(nn.Module):
    """
    Two-sided encoder = Transformer (Attention)
    Transformer = Multi-head Attention + feed forward network, connected with sublayers
    """
    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        hidden: transformer hidden layer size 
        attn_heads: multi-attention heads 
        feed_forward_hidden: feed forward network hidden layer size, normally 4 * hidden_size 
        dropout
        """
        super().__init__()
        self.attention = MultiHeadAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask):
        # apply attention to input x, connect with input sublayer
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        # apply feedforward to x, connect with output sublayer 
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # calculate position encoding. full 0 tensor to store position encoding.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False # no need to gradient, as position encoding is fixed, no training needed
        # position tensor, from 0 to max_len - 1 
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term) # even index, use sin 
        pe[:, 1::2] = torch.cos(position * div_term) # odd index, use cos
        pe = pe.unsqueeze(0) # add one dim, to match input data 
        self.register_buffer('pe', pe) # register position encoding as a buffer, it can move with model, not seen as model param

    def forward(self, x):
        return self.pe[:, :x.size(1)] # return pe matching input sequence length 
    
class BERT(nn.Module):
    """
    BERT, with Transformer two-sided encoder
    """
    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """
        vocab_size
        hidden: BERT hidden layer size = 768
        n_layers: Transformer blocks size = 12 
        attn_heads = 12 
        dropout = 0.1
        """
        super().__init__()
        self.hidden = hidden 
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        # 4 * hidden_size as feed forward network hidden layer size 
        self.feed_forward_hidden = hidden * 4 
        # BERT embedding, including position embedding, word embedding
        self.word_embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden)
        self.position_embedding = PositionalEmbedding(d_model=hidden)
        # multiple Transformer blocks, deep network 
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)]
        ) # multiple Transformer blocks

    def forward(self, x):
        """
        Feed Forward
        x: input sequence, shape [batch_size, seq_len]
        after BERT, output shape [batch_size, seq_len, hidden]
        """
        # attention mask for token creation 
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # index sequence embed to vector sequence, sum up word embedding and positionn embedding as input sequence embedding 
        x = self.word_embedding(x) + self.position_embedding(x)
        # for multiple Transformer blocks 
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask) # input sequence, attention mask -> all Transformer blocks
        return x # output after all Transformer blocks
    

if __name__ == '__main__':
    vocab_size = 1024 
    seq = arr = torch.tensor([[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]])
    logits = BERT(vocab_size=vocab_size)(seq)
    print(logits.shape)
```

<!-- TOC --><a name="4-attention-2rope"></a>
## 4. Attention 2：RoPE

旋转位置编码RoPE：构建位置相关的投影矩阵，让Q和K在计算时达到平衡

```py
class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, scale_base=model_config.scale_base, use_xpos=True):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.use_xpos = use_xpos 
        self.scale_base = scale_base 
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device=all_config.device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.enisum('i, j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim=-1)
        if not self.use_xpos:
            return freqs, torch.ones(1, device=device)
        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** elt.Rearrange('n -< n 1')(power) # rearrange (power, )
        scale = torch.cat((scale, scale), dim=-1)
        return freqs, scale 
    
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    def apply_rotary_pos_emb(pos, t, scale=1.):
        return (t * pos.cos() * scale) + (rotate_half(t) * pos.sin() * scale)
    
if __name__ == '__main__':
    embedding = torch.randn(size=(5, 128, 512))
    print(rotate_half(embedding).shape)
```

SwiGLU (Swish-Gated Linear Unit)：基于门控机制的激活函数，可以捕捉序列长依赖关系。
- GLU用sigmoid激活函数把信号转换为0~1值（表示重要性），乘以门控值来选择性放大或抑制输入，根据上下文选择性关注某些句子语义。
- Swish：类似ReLU的非线性函数 = x * sigmoid(x)，输入正数时趋向于线性变换，负数时有非线性抑制效果。

```py
class SwiGLU(torch.nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return torch.nn.functional.silu(gate) * x
```

因果掩码causal mask：将当前token后所有内容掩码，让这些信息不参与损失函数计算，防止模型预测使用未来信息。

Case Study
- Hotel comments sentiment analysis. Clean code (P69 - 77).
- 自回归文本生成模型：Block模块化设计思想，堆叠多个Block对文本特征逐层抽取，并在logits层进行转换后输出。

<!-- TOC --><a name="5-attention-3moe"></a>
## 5. Attention 3：MoE

稀疏MoE = 专家 + 路由（调度员：输入数据与路由权重矩阵乘法，得出专家适配的得分，然后激活一部分专家来处理，将这些专家的输出与其对应的门控得分相乘进行加权，合并输出）
- Dense MoE考虑所有专家，但根据不同权重选择各专家，可以全面整合各专家意见，适合复杂任务。Sparse MoE只激活少数专家，适合效率提升
- KeepTopK将每个token路由到选定专家（Token Choice）。而且可扩展，系统高峰时自动增加K值，容纳更多节点；低峰期减小K值，节省资源。

```py
import torch

# Expert is a fully connected network
class Expert(torch.nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_embd, 4 * n_embd) # linear layer, dim * 4 
            torch.nn.ReLU(),
            torch.nn.Linear(4 * n_embd, n_embd) # linear layer, dim return to original
            torch.nn.Dropout(0.1), # dropout to avoid overfitting
        )

    def forward(self, x):
        return self.net(x) 
    
# select top k experts
class TopkRouter(torch.nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k 
        self.linear = torch.nn.Linear(n_embd, num_experts) # linear layer, output num of experts

    def forward(self, mh_output):
        logits = self.linear(mh_output) # linear layer -> each expert score 
        # select top k experts and their indices 
        top_k_logits, indices = logits.topk(self.top_k, dim=-1)
        # tensor with same shape of logits, all -inf values 
        zeros = torch.full_like(logits, float('-inf'))
        # k experts scores -> zeros with according locations
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        # softmax for sparse_logits, scores -> prob distribution 
        router_output = torch.nn.functional.softmax(sparse_logits, dim=-1)
        return router_output, indices 
    
class SparseMoE(torch.nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(SparseMoE, self).__init__()
        # router to select experts
        self.router = TopkRouter(n_embd, num_experts, top_k)
        # a list of experts
        self.experts = torch.nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.top_k = top_k 

    def forward(self, x):
        # router -> expert prob distribution and indices
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x) # initialize outputs with all-0 tensor
        # flatten input and router's output, for convenience 
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # for each expert, process input based on their prob distribution 
        for i, expert in enumerate(self.experts):
            # find k-th expert token
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1) # flatten 
            # if current expert has at least one token being one of top-k experts 
            if flat_mask.any():
                # select these tokens' input 
                expert_input = flat_x[flat_mask]
                # given these token inputs to current expert to work on 
                expert_output = expert(expert_input)
                # current expert prob distribution for these tokens
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                # weighted average based on experts' prob distribution 
                weighted_output = expert_output * gating_scores
                # cumulate weighted scores, then output to corresponding locations 
                final_output[expert_mask] += weighted_output.squeeze(1)
        return final_output 
```

Case Study：把上一章情感分类任务的注意力层换成MoE层，相比单一FFN可以更好捕捉复杂性（很多专家竞争机制），而且高效（不用每次前向传播都激活所有神经元），还可扩展（根据数据量增减专家数量，无需从头训练整个网络）。不同token可能被不同专家处理，导致模型内部形成多样化路径，这给模型提供了灵活的动态特征选择机制，在面对不同输入时可以精准聚焦关键信息、调整关注重点，提高泛化和复杂任务能力。


修正MoE门控函数：避免所有token都集中在热门expert，让token分配不集中也不分散 -> 在门控线性层logits上添加标准正态噪声。噪声的随机性让token在选expert时不再完全依赖原始logits，可以打破热门expert垄断地位，提高鲁棒性并防止模型训练过早收敛/陷入局部最优。

```py
class NoisyTopkRouter(torch.nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k 
        self.topkroute_linear = torch.nn.Linear(n_embed, num_experts)
        # add noise 
        self.noise_linear = torch.nn.Linear(n_embed, num_experts)

    def forward(self, mh_output):
        # mh_output: output tensor from multihead self attention block 
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)
        # add scaled unit gaussian noise to logits 
        noise = torch.randn_like(logits) * torch.nn.functional.softplus(noise_logits)
        noisy_logits = logits + noise 
        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = torch.nn.functional.softmax(sparse_logits, dim=-1)
        return router_output, indices 
```

图像分类：通道注意力机制（CNN每个卷积层都输出feature maps，每个特征图对应一个channel，每个通道对最终决策的贡献是相同的，而Squeeze-and-Excitation Block可以通过Squeeze & Excitation对通道关系特征重新校准，提高性能）。
- Squeeze：把全局空间信息压缩到一个通道描述符，用Global Average Pooling (GAP)
- Excitation：Squeeze得到的通道描述符，学习通道间相互依赖关系，给每个通道生成权重，用两个全连接层+激活函数实现。

ViT (Vision Transformer)：把图像分成小块Patch（一个token是一个矩阵），这些小块的线性映射序列作为自注意力模块传入网络，可以长距离依赖、可解释性、并行计算、全局感知、易于迁移学习。
- V-MoE：把ViT密集的FFNN层替换为稀疏的MoE层，提高效率和性能。

```py
class VIT(torch.nn.Module):
    def __init__(self, dim=312):
        super(VIT, self).__init__()
        self.patch_embedding = PatchEmbed()
        self.position_embedding = torch.nn.Parameter(torch.rand(1, 16, dim))
        self.vit_layers = VITBlock(d_model=312, num_heads=6)
        self.logits_layer = torch.nn.Linear(4992,10)
    def forward(self, x):
        embedding = self.patch_embedding(x) + self.position_embedding
        embedding = self.vit_layers(embedding)
        embedding = torch.nn.Flatten()(embedding)
        logits = self.logits_layer(embedding)
        return logits 
    
# pip install st_moe_pytorch
import torch 
from st_moe_pytorch import MoE
from st_moe_pytorch import SparseMoEBlock 

class MOE(torch.nn.Module):
    def __init__(self, dim=512):
        super(MOE, self).__init__()
        self.moe = MoE(
            dim=dim,
            num_experts = 16,
            gating_top_n = 2,
            # decide a token is routed to 2nd experts or after. For 2 experts, 0.2 is the best threshold
            threshold_train = 0.2,
            threshold_eval = 0.2,
            # some extra capacity to avoid unbalanced routing 
            capacity_factor_train = 1.25,
            # capacity_factor_* >= 1 
            capacity_factor_eval = 2.,
            balance_loss_coef = 1e-2, # aux experts coef for balance loss
            router_z_loss_coef = 1e-3,
        )
        self.moe_block = SparseMoEBlock(
            self.moe,
            add_ff_before=True,
            add_ff_after=True
        )
        self.norm = torch.nn.RMSNorm(dim)
        self.moe_linear = torch.nn.Linear(dim, dim, bias=False)
        self.activity_layer = Swiglu(hidden_size = dim)

    def forward(self, x):
        x = self.norm(x)
        enc_out = self.moe_block(x)[0]
        enc_out = self.activity_layer(enc_out) # torch.nn.functional.gelu(enc_out)
        enc_out = self.moe_linear(enc_out)
        return enc_out
```

<!-- TOC --><a name="6-attention-4mla"></a>
## 6. Attention 4：MQA, MLA, GQA

MQA (Multi-Query Attention) 多查询注意力：传统MHA中QKV根据每个头进行不同变换，但头数量众多时导致计算量太大。MQA增强关键信息捕捉能力，适合复杂任务，让所有头共享同一组KV矩阵，减少计算量和参数量，只有Q保留多头特性，但会牺牲精度。

```py
import torch 
import torch.nn as nn 
import einops 

class MultiHeadAttention_MQA(torch.nn.Module):
    def __init__(self, d_model, attention_head_num):
        super(MultiHeadAttention_MQA, self).__init__()
        self.attention_head_num = attention_head_num
        self.d_model = d_model 

        assert d_model % attention_head_num == 0 
        self.scale = d_model ** -0.5 # scale attention score 
        self.per_head_dmodel = d_model // attention_head_num # each attention head dim 
        # linear layer for QKV, note that K and V dim are reduced 
        self.qkv_layer = torch.nn.Linear(d_model, (d_model + 2 * self.per_head_dmodel))
        # rotary embedding layer, to rotate Q, K 
        self.rotary_embedding = RotaryEmbedding(self.per_head_dmodel // 2, use_xpos=True)
        self.out_layer = torch.nn.Linear(d_model, d_model) # output linear layer 

    def forward(self, embedding, past_length=0):
        B, S, D = embdding.shape # tensor shape: batch size, sequence length, dim 
        # linear layer for QKV 
        qky_x = self.qkv_layer(embedding)
        q, k, v = torch.split(qky_x, [self.d_model, self.per_head_dmodel, self.per_head_dmodel], dim=-1)
        # rearrange Q, so each attention head can handle its work independently 
        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.attention_head_num)
        # K, V dim expansion, to broadcast with Q, this is the important part for MQA to share K and V 
        k = k.unsqueeze(2).expand(B, -1, self.attention_head_num, -1).transpose(1, 2)
        v = v.unsqueeze(2).expand(B, -1, self.attention_head_num, -1).transpose(1, 2)

        # rotary embdding to rate Q and K 
        q, k = self.rotary_embedding.rotate_queries_and_keys(q, k, seq_dim=2)
        q = q * self.scale 
        # attention score 
        sim = eniops.einsum(q, k, 'b h i d, b h j d -> b h i j')

        # causal mask to mask future position 
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool).triu(past_length).to(embedding.device)
        # use big negative number to mask future position 
        sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
        attn = sim.softmax(dim=-1)
        # attention weight to get weighted sum for V 
        out = einops.einsum(attn, v, 'b h i j, b h j d -> b h i d')
        # rearrange output side to match input size 
        embedding = einops.rearrange(out, "b h s d -> b s (h d)")
        embedding = self.out_layer(embedding) 
        return embedding
```

MLA用低秩键值联合压缩QKV到更低维度的潜空间，提高效率，优化模型内存和计算速度。

```py
class MultiHeadAttention_MLA(torch.nn.Module):
    def __init__(self, d_model, attention_head_num):
        super(MultiHeadAttention_MLA, self).__init__()
        self.attention_head_num = attention_head_num 
        self.d_model = d_model 
        assert d_model % attention_head_num == 0 
        self.scale = d_model ** -0.5
        self.softcap_value = 50.
        self.per_head_dmodel = d_model // attention_head_num # each attention head dim 

        # Q linear layer and normalization layer
        self.q_rope_dense = torch.nn.Linear(self.per_head_dmodel, self.per_head_dmodel * 2)
        self.q_norm = torch.nn.RMSNorm(self.per_head_dmodel * 2)
        # QK dim in low-rank latent space
        self.qk_nope_dim = self.per_head_dmodel
        self.qk_rope_dim = self.per_head_dmodel 
        # KV projection dim and relevant layer 
        self.kv_proj_dim = self.d_model 
        self.kv_proj_dim_VS_qk_rope_dim = (self.kv_proj_dim + self.qk_rope_dim)
        self.kv_layernorm = torch.nn.RMSNorm(self.kv_proj_dim)
        self.kv_dense = torch.nn.Linear(self.kv_proj_dim, (self.d_model + self.attention_head_num * self.qk_nope_dim))

        # linear layer for QKV initial representation 
        self.qkv_layer = torch.nn.Linear(d_model, (d_model + self.kv_proj_dim_VS_qk_rope_dim))
        self.rotary_embedding = RotaryEmbedding(self.per_head_dmodel // 2)
        self.out_layer = torch.nn.Linear(d_model, d_model)

    def forward(self, embedding, past_length=0):
        B, S, D = embedding.shape
        # get initial representation for QKV by linear layer
        qky_x = self.qkv_layer(embedding)
        # split q and compressed kv
        q, compressed_kv = torch.split(qky_x, split_size_or_sections=[self.d_model, self.kv_proj_dim_VS_qk_rope_dim], dim=-1)
        # rearrange Q and linear transformation, normalization 
        q = einops.rearrange(q, "b s (h d) -> b h s d", h=self.attention_head_num)
        q = self.q_norm(self.q_rope_dense(q))

        # separate Q to 2 parts, apply rotary embedding for one part 
        q, q_for_rope = torch.split(q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1)
        q_for_rope = self.rotary_embedding.rotate_queries_or_keys(q_for_rope)

        # split compressed KV, normalization and linear transformation 
        KV_for_lora, K_for_rope = torch.split(compressed_kv, [self.kv_proj_dim, self.qk_rope_dim], dim=-1)
        KV_for_lora = self.kv_layernorm(KV_for_lora)
        KV = self.kv_dense(KV_for_lora)
        KV = einops.rearrange(KV, "b s (h d) -> b h s d", h=self.attention_head_num)
        K, V = torch.split(KV, [self.qk_nope_dim, self.per_head_dmodel], dim=-1)

        # expand K_for_rope to match attention head size 
        K_for_rope = einops.repeat(K_for_rope, "b s d -> b h s d", h=self.attention_head_num)
        # combine Q, K heads for attention score calculation 
        q_heads = torch.cat([q, q_for_rope], dim=-1)
        k_heads = torch.cat([K, K_for_rope], dim=-1)
        v_heads = V # has been rearranged previously

        # scale Q for attention score calculation
        q_heads = q_heads * self.scale 
        sim = einops.einsum(q_heads, k_heads, 'b h i d, b h j d -> b h i j')
        # causal mask, calculate softmax attention weight 
        mask_value = -torch.finfo(sim.dtype).max 
        i, j = sim.shape[-2:]
        causal_mask = torch.ones((i, j), dtype=torch.bool).triu(past_length).to(embedding.device)
        sim = sim.masked_fill(causal_mask, mask_value)
        attn = sim.softmax(dim=-1)

        # attention weight for V, get embedding 
        out = einops.einsum(attn, V_heads, 'b h i j, b h j d -> b h i d')
        embedding = einops.rearrange(out, "b h s d -> b s (h d)")
        embedding = self.out_layer(embedding)
        return embedding
```

GQA分组查询注意力：全面优化，不仅改进了注意力计算方式，还引入全局信息，提高整体理解能力。查询头被分成G组，每个查询头都有独立的参数空间，每个组共享一套KV矩阵，可以保持注意力机制灵活性和参数高效利用。GQA比MQA（多头查询注意力）有更高插值模型质量且速度更快，H个键值头缩减为1个键值头，数据量减小H倍。（代码太长，略）

Multi Head Differential Attention差分注意力机制：对于长文本，Transformer会注意力分散（关注与任务无关内容，注意力噪声干扰了对关键信息的捕捉），因此在注意力得分计算时引入差分思想，计算两个独立softmax注意力图之差，滤除噪声，强化对有效信号的捕捉。（代码太长，略）

Case Study：基于MLA的语音情感分类。

<!-- TOC --><a name="9-diffusion"></a>
## 9. Diffusion

Diffusion using UNet as generation model, from noise to generate image. UNet对训练数据要求少，因为大量参数共享和特征重用的结构，可扩展性（到不同尺寸）、适应性强（不同图像分割任务）。

UNet：预测每个图片所添加的噪声，通过计算的方式去除对应的噪声，重现原始图像。
- 初始化：卷积层`init_conv`、时间嵌入`time_mlp`、下采样`downs`、中间模块`mid_block1`, `mid_attn`, `mid_block2`、上采样`ups`

```py
import torch 
import get_dataset 
import cv2
from tqdm import tqdm 
import ddpm 

batch_size = 48 
dataloader = torch.utils.data.DataLoader(get_dataset.SamplerDataset(), batch_size=batch_size)

# Unet as generative model, from noise to image generation
import unet 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = unet.Unet(dim=28, dim_mults=[1, 2, 4]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

epochs = 3 
timesteps = 200 
save_path = "./saver/ddpm_saver.pth"
model.load_state_dict(torch.load(save_path), strict=False)
for epoch in range(epochs):
    pbar = tqdm(dataloader, total=len(dataloader))
    # use DataLoader for iteration, get each batch data sample
    for batch_sample, batch_label in pbar:
        optimizer.zero_grad()
        batch_size = batch_sample.size()[0]
        batch = batch_sample.to(device)
        optimizer.zero_grad()
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = ddpm.p_losses(model, batch, t, loss_type="huber")
        loss.backward()
        optimizer.step()
        pbar.set_description(f"epoch:{epoch + 1}, train_loss: {loss.item():.5f}")
    torch.save(model.state_dict(), save_path)
```



<!-- TOC --><a name="10-multimodal-fusion"></a>
## 10. Multimodal Fusion

<!-- TOC --><a name="11-cross-attention-audio"></a>
## 11. Cross-Attention, Audio

<!-- TOC --><a name="12-token-compression"></a>
## 12. Token Compression

<!-- TOC --><a name="13-image-encoder-vq-vae-fsq"></a>
## 13. Image Encoder: VQ-VAE, FSQ

<!-- TOC --><a name="14-torchvision-video-classification"></a>
## 14. torchvision Video Classification

<!-- TOC --><a name="15-deepseek-agents"></a>
## 15. DeepSeek Agents
