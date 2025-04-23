DeepSeek Large Model High-Performance Core Technology and Multimodal Fusion Development, by Xiaohua Wang, 2025

<img src="https://github.com/user-attachments/assets/1c3c59ff-4b22-4163-a12b-cfc027950602" width="32%" height="32%">

## 3. 注意力1

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

## 4. 注意力2

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









