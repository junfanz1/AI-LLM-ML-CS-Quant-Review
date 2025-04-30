LLM from Theory to Practice, by Qi Zhang, 2024

<img src="https://github.com/user-attachments/assets/58741520-5516-4c3c-8a65-26f2e158a5d9" width="32%" height="32%">

# Contents

<!-- TOC start (generated with https://github.com/derlin/bitdowntoc) -->

- [2. LLM Intro](#2-llm-intro)
- [4.Distributed Training](#4distributed-training)
- [5. SFT](#5-sft)
- [6. RL](#6-rl)
- [7. Applications ](#7-applications)
- [8. Evaluations](#8-evaluations)

<!-- TOC end -->

<!-- TOC --><a name="2-llm-intro"></a>
## 2. LLM Intro

```py
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model 

    # const PE matrix by pos and i 
    pe = torch.zeros(max_seq_len, d_model)
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            pe[pos, i] = math.sin(pos / (10000 ** (i / d_model)))
            pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d_model)))
    
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Word embedding make a bit bigger 
        x = x * math.sqrt(self.d_model)
        # add position const to word embedding 
        seq_len = x.size(1)
        x = x + Variable(self.pe[:, :seq_len], requires_grad=False).cuda()
        return x 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model 
        self.d_k = d_model // heads 
        self.h = heads 

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        # mask those added units for filling length, so that after softmax = 0
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)

        output = torch.matmul(scores, v)
        return output 
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # linear, divide by k heads 
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concat multiple heads and output to final linear layer 
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)

        return output 
    
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x 
    
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model 

        # layer normalization includes two learnable params 
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps 

    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
            / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm 
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attn(x, x, x, mask)
        attn_output = self.dropout_1(attn_output)
        x = x + attn_output
        x = self.norm_1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = x + ff_output
        x = self.norm_2(x)
        return x 
    
class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N 
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layer = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, src, mask):
        x = self.embed(src)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask, trg_mask):
        attn_output_1 = self.attn_1(x, x, x, trg_mask)
        attn_output_1 = self.dropout_1(attn_output_1)
        x = x + attn_output_1
        x = self.norm_1(x)
        attn_output_2 = self.attn_2(x, e_outputs, e_outputs, src_mask)
        attn_output_2 = self.dropout_2(attn_output_2)
        x = x + attn_output_2 
        x = self.norm_2(x)

        ff_output = self.ff(x)
        ff_output = self.dropout_3(ff_output)
        x = x + ff_output
        x = self.norm_3(x)

        return x 
    
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N 
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, trg, e_outputs, src_mask, trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x)
    
class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, src, trg, src_mask, trg_mask):
        e_outputs = self.encoder(src, src_mask)
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        output = self.out(d_output)
        return output 
    
d_model = 512
heads = 8 
N = 6
src_vocab = len(EN_TEXT.vocab)
trg_vocab = len(FR_TEXT.vocab)
model = Transformer(src_vocab, trg_vocab, d_model, N, heads)
for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optim = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

def train_model(epochs, print_every=100):
    model.train()
    start = time.time()
    temp = start 
    total_loss = 0 
    for epoch in range(epochs):
        for i, batch in enumerate(train_iter):
            src = batch.English.transpose(0, 1)
            trg = batch.French.transpose(0, 1)
            # translate English words to French, except the last word, no need for next prediction as it's end 
            trg_input = trg[:, :-1]
            # predict words 
            targets = trg[:, 1:].contiguous().view(-1)
            # use mask creation methods
            src_mask, trg_mask = create_masks(src, trg_input)
            preds = model(src, trg_input, src_mask, trg_mask)

            optim.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)),
                                   results, ignore_index = target_pad)
            loss.backward()
            optim.step()

            total_loss += loss.data[0]
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time =%dm, epoch %d, loss = %.3f, %ds per %d iters" % ((time.time() - start) // 60,
                                                                              epoch + 1, i + 1, loss_avg, time.time() - temp, print_every))
                total_loss = 0 
                temp = time.time()

    # model testing 
    def translate(model, src, max_len = 80, custom_string=False):
        model.eval()
        if custom_sentence == True:
            src = tokenize_en(src)
            sentence = Variable(torch.LongTensor([[EN_TEXT.vocab.stoi[tok] for tok in sentence]])).cuda()
        src_mask = (src != input_pad).unsqueeze(-2)
        e_outputs = model.encoder(src, src_mask)
        outputs = torch.zeros(max_len).type_as(src.data)
        outputs[0] = torch.LongTensor([FR_TEXT.vocab.stoi['<sos>']])

        for i in range(1, max_len):
            trg_mask = np.triu(np.ones((1, i, i), k=1).astype('uint8'))
            trg_mask = Variable(torch.from_numpy(trg_mask) == 0).cuda()
            out = model.out(model.decoder(outputs[:i].unsqueeze(0), e_outputs, src_mask, trg_mask))
            out = F.softmax(out, dim=-1)
            val, ix = out[:, -1].data.topk(1)
            outputs[i] = ix[0][0]
            if ix[0][0] == FR_TEXT.vocab.stoi['<eos>']:
                break 
        return ' '.join([FR_TEXT.vocab.itos[ix] for ix in outputs[:i]])
```

Llama
- RMSNorm归一化

```py
class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps # eps to avoid divided by 0 

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype 
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # weight can be multiplied by trainable parameter, g_i
        return (self.weight * hidden_states).to(input_dtype)
```

- SwiGLU
- RoPE：外推能力不好，可以用ALiBi让模型有更长上下文建模能力

```py
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # `torch.jit,trace` to work
        self.max_seq_len_cached = max_position_embeddings 
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # __init__ construct sin/cos, this if is not possible to execute
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = torch.arange(self.max_seq_len_cached, device=x.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", emb.cos()[None, None, :, :].to(x.dtype), persistent=False)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )
    
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2] # another half input use rotate to hide dimension
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
        # cos and sin first two dimensions are always 1, can do squeeze 
        cos = cos.squeeze(1).squeeze(0) # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0) # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1) # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1) # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed        
```

```py
class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size 
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # self attention 
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states 

        # fully connected layer 
        residual = hidden_states 
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states 

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs
```

注意力机制优化
- 稀疏注意力：Global Attention（加入全局节点）、Band Attention（大部分数据具有局部性，限制Query只与相邻节点交互）、Dilated Attention（类似CNN的Dilated Conv，通过增加空隙获得更大感受野）、Random Attention（随机采样、提升非局部交互能力）、Block Local Attention（用多个不重叠Block限制信息交互）
    - Star-Transformer = 带状注意力（宽度=3）+全局注意力（任意两个非相邻节点通过共享的全局注意力连接，相邻节点直接连接）
    - Longformer = 带状注意力 + Internal Global-node Attention，上层一些带状注意力头部替换为具有膨胀窗口的注意力，在增加感受野的同时不增加计算量
    - Extended Transformer Construction = 带状注意力 + External Global-node Attention。稀疏注意力包括掩码机制处理结构化输入，用Contrastive Predictive Coding预训练
    - BigBird = 带状注意力 + 全局注意力，用额外随机注意力近似全连接注意力，且稀疏编码器和稀疏解码器可以模拟任何图灵机，因此稀疏注意力模型很好。
    - Routing Transformer：用K-Means聚类，对QK聚类，每个Q只与其在相同cluster下的K交互，中心向量用滑动平均来更新。
    - Reformer：Local-Sensitive Hashing对每个Q选择KV对，把QK进行哈希计算并划分到多个桶里，提高同一个桶QK参与交互的概率。
- Flash Attention，`torch.backends.cuda.enable_flash_sdp()`
    - 用GPU硬件特殊设计，对全局内存和共享存储IO速度不同，避免从High Bandwidth Memory（全局内存）读取或写入注意力矩阵，尽可能高效用Shared Memory加快计算速度。
    - 要在不访问整个输入的情况下计算softmax，后向传播中不存储中间注意力矩阵（分块写入，在输入块上多次传递，增量方式算softmax），通过存储归一化因子来减少全局内存消耗
- Multi-Query Attention：不同注意力头共享一个键值对，因此键值矩阵只有一份， 减少显存占用。

```py
class MultiQueryAttention(nn.Module):
    # use torch or triton for attention, allow for additional shift 
    def __init__(
            self,
            d_model: int,
            n_heads: int, 
            device: Optional[str] = None,
    ):
        super().__init__()
        self.d_model = d_model 
        self.n_heads = n_heads 
        self.head_dim = d_model // n_heads 

        self.Wqkv = nn.Linear(
            # create Multi Query Attention 
            d_model, 
            d_model + 2 * self.head_dim, # only create query head vector, so only 1 d_model 
            device=device, # KV not using unique head vectors
        )
        self.attn_fn = scaled_multihead_dot_product_attention 
        self.out_proj = nn.Linear(
            self.d_model,
            self.d_model,
            device=device
        )
        self.out_proj._is_residual = True 

    def forward(self, x):
        qkv = self.Wqkv(x) # (1, 512, 960)
        query, key, value = qkv.split(
            # query -> (1, 512, 768), key -> (1, 512, 96), value -> (1, 512, 96)
            [self.d_model, self.head_dim, self.head_dim],
            dim=2
        )
        context, attn_weights, past_key_value = self.attn_fn(
            query, key, value, self.n_heads, multiquery=True
        )
        return self.out_proj(context), attn_weights, past_key_value
```

<!-- TOC --><a name="4distributed-training"></a>
## 4.Distributed Training

- Mini-batch：数据小批次根据损失函数和优化算法计算梯度，修正模型参数。
- 数据并行：每个设备只分配一个批次数据样本的子集。

```py
class DistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset 
        self.num_replicas = num_replicas # threads number, by default = wolrd_size (GPU devices)
        self.rank = rank # current which thread/GPU 
        self.epoch = 0 
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas)) # each thread sample numbers 
        self.total_size = self.num_samples * self.num_replicas # total sample numbers in dataset 
        self.shuffle = shuffle # if shuffle dataset or not 
        self.seed = seed 

    def __iter__(self):
        # 1. shuffle dataset order 
        if self.shuffle:
            # based on training rounds and seeds number
            g = torch.Generator()
            # self.seed is fixed, by set_epoch we can change initializing seeds to change self.epoch 
            # can shuffle order in each epoch training, let each round each GPU get different data for better training 
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # data augmentation 
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # assign data 
        indices = indices[self.rank: self.total_size: self.num_replicas]
        assert len(indices) == self.num_samples 
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch
```

- 模型并行
    - 按层切分：流水线并行。为了减少并行气泡，可以用GPipe，将Mini-batch划分成更小的Micro-batch。Megatron-LM用1F1B非交错式调度流水线策略（一个前向通道一个后向通道，比GPipe在内存节省方面更好）。
    - 计算图层内参数切分：张量并行。FFN有两层全连接层，可以把多头注意力机制的两个矩阵乘切分，张量并行。PyTorch有细粒度张量并行API `DistributedTensor`对大张量分片。

```py
import torch
from torch.distributed._tensor import DTensor, DeviceMesh, Shard, distribute_tensor, distribute_module

device_mesh = DeviceMesh("cuda", [0, 1, 2, 3])
rowwise_placement = [Shard(0)]
colwise_placement = [Shard(1)]
big_tensor = torch.randn(888, 12)
rowwise_tensor = distribute_tensor(big_tensor, device_mesh=device_mesh, placements=rowwise_placement)

class MyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 8)
        self.fc2 = nn.Linear(8, 8)
        self.relu = nn.ReLU()

    def forward(self, input):
        return self.relu(self.fc1(input) + self.fc2(input))
    
mesh = DeviceMesh(device_type="cuda", mesh=[[0, 1], [2, 3]])

def shard_params(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    def to_dist_tensor(t): return distribute_tensor(t, mesh, rowwise_placement)
    mod._apply(to_dist_tensor)

sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_params)

def shard_fc(mod_name, mod, mesh):
    rowwise_placement = [Shard(0)]
    if mod_name == "fc1":
        mod.weight = torch.nn.Parameter(distribute_tensor(mod.weight, mesh, rowwise_placement))

sharded_module = distribute_module(MyModule(), mesh, partition_fn=shard_fc)
```

 - Adam优化器：需要计算一阶Momentum和二阶Variance，内存占用大，可以用Dynamic Loss Scaling, Mixed Precision Optimizer
 - 混合并行：Megatron-LM提供张量并行；DeepSpeed提供ZeRO零冗余优化器（降低显存占用，对模型状态的存储去除冗余，用分区的方法把模型状态量分割，每个设备只保存一部分）、模型流水线和分布式训练组件。PyTorch用`torch.nn.parallel.DistributedDataParallel`.
 - 集群架构：分布式训练计算集群的拓扑结构是multi-level tree，用Top of Rack Switch连接网络，增加spine switch接入新机柜。由于cross-rack communication瓶颈，用Fat-Tree拓扑结构实现网络带宽无收敛。
 - 参数服务器Parameter Server架构 = 训练服务器 + 参数服务器（提供内存和通信）。异步训练：训练服务器完成小批次训练后，将梯度推给参数服务器，参数服务器不再等待接收所有梯度，直接基于已收到的梯度进行参数更新。
 - 去中心化架构：用Collective Communication 集合通信实现分布式训练，通信原语包括Broadcast, Scatter, Reduce, All Reduce, Gather, All Gather, Reduce Scatter, All to All.
     - `torch.distributed`初始化分布式环境，`torch.multiprocessing`开启多进程
 - DeepSpeed：灵活组合三种并行（ZeRO支持的数据并行、流水线并行、张量并行），可处理万亿参数超大模型。提供Sparse Attention Kernel可处理长序列。集成1-bit Adam，只用Adam算法1/5的通信量，达到类似收敛率。

LLaMA分布式训练 + DeepSpeed

```py
# 1. Training data config

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler 
from transformers import default_data_collector 
from utils.data.data_utils import create_pretrain_dataset 

# data prep
train_dataset, eval_dataset = create_pretrain_dataset(
    args.local_rank, args.data_path, args.data_split, args.data_output_path,
    args.seed, tokenizer, args.max_seq_len
)

# create DataLoader 
if args.local_rank == -1:
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
else:
    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset)
train_dataloader = DataLoader(train_dataset, collate_fn = default_data_collector,
                              sampler = train_sampler, batch_size = args.per_device_train_batch_size)
eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collector,
                             sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)

# 2. Load LLaMA model from transformers, use from_pretrained to load pretrained model

from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig 

# use tokenizer to get correct tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, fast_tokenizer=True)
if tokenizer.pad_token is None:
    # check tokenizer.eos_token not None, add special token in tokenizer 
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'

model_config = LlamaConfig.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=model_config)
model.config.end_token_id = tokenizer.eos_token_id 
model.config.pad_token_id = model.config.eos_token_id 
model.resize_token_embeddings(int(8 * math.ceil(len(tokenizer) / 8.0))) # token embedding size can be divided by 8 for performance optimization on hardware

# 3. Optimization to enhance training speed
"""
- params split to two groups (weight decay, and none, for regularization and avoid overfitting)
- DeepSpeedCPUAdam or FusedAdam
- Learning rate warmup, dynamic adjustments
"""

from transformers import get_scheduler 
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam 
# set up optimizer and model params 
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model, args.weight_decay, args.learning_rate
)
AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.95))
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps
)
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
)

def get_optimizer_grouped_parameters(model, weight_decay, no_decay_name_list=["bias", "LayerNorm.weight"]):
    # weights split to 2 groups, one with weight decay, another none 
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                       if (not any (nd in n for nd in no_decay_name_list) and p.requires_grad)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                       if (any (nd in n for nd in no_decay_name_list) and p.requires_grad)],
            "weight_decay": 0.0,
        }
    ]
    return optimizer_grouped_parameters

# 4. DeepSpeed config
"""
- ZeRO config, to reduce redundancy and enhance speed,
- Mixed precision, FP16
- gradient_clipping to avoid gradient explosion 
- hybrid_engine
- TensorBoard config to track training process
- get_eval_ds_config: eval set
"""

# 5. DeepSpeed initialization

import deepspeed 
if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # initialize distributed backend, synchronize GPU
    torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()
args.global_rank = torch.distributed.get_rank()

ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage,
                                enable_tensorboard=args.enable_tensorboard,
                                tb_path=args.tensorboard_path, tb_name="step1_model")
ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size 
ds_config['train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size() * args.gradient_accumulation_steps

set_random_seed(args.seed)
torch.distributed.barrier()

# initialize optimizer and model with DeepSpeed 
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model, optimizer=optimizer, args=args, config=ds_config,
    lr_scheduler=lr_scheduler, dist_init_required=True
)
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()

# 6. Model Training
"""
- Before training, evaluate model, calculate perplexity
- Training iteration: model.backward(loss) to calculate gradient, model.step() to update parameters.
    For main thread, print_throughput to know model training speed
- Save model as HuggingFace format
"""
```

<!-- TOC --><a name="5-sft"></a>
## 5. SFT

- LoRA
    - AdaLoRA：根据下游任务重要性调整秩的大小
    - QLoRA：新数据类型NF4、双重量化、分页优化器（显存不足时自动将优化器状态转移至内存） 

```py
from transformers import AutoModelForSeq2SeqLM 
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType 
model_name_or_path = ""
tokenizer_name_or_path = ""
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_config(model, peft_config)
```

- DeepSpeed Chat训练对话模型RLHF系统三步骤
    - 人类标记数据 + 预训练模型 = 有监督微调
    - 好坏回答对 + 预训练模型 = 奖励模型
    - 有监督微调 -> 演员模型、冻结参数的参考模型、EMA；奖励模型 -> 评论模型、冻结参数的奖励函数。用演员模型生成输入，把前面这些都放入PPO
    - 训练、推理能力整合到混合引擎，用于RLHF训练，让DeepSpeed无缝在推理和训练模式间切换。

<!-- TOC --><a name="6-rl"></a>
## 6. RL

- Actor-Critic Agent：既学习Policy有学习Value function，通过两者交互得到最佳动作。
- PPO流程
    - 环境采样：策略模型对给定输入生成回复，奖励模型对回复打分获得奖励。
    - 优势估计：用评论模型预测生成回复的未来累计奖励，用Generalized Advantage Estimation (GAE)（k步优势的指数平均，k增大则时序差分趋向于蒙特卡洛）估计优势函数，更准确评估每次行动的好处。
    - 优化调整：用优势函数优化调整策略模型，用参考模型确保更新的策略变化不大，稳定性好。
- 模型训练：奖励模型基于Transformer的预训练语言模型，移除最后一个非嵌入层，叠加一个额外线性层，奖励模型能给最后一个token分配标量奖励值。
    - 模仿学习：训练数据是专家正确答案。奖励函数的附加项：基于学习得到强化学习策略与初始监督模型的KL散度（熵奖励可以促进策略空间探索、避免策略过早收敛到单一模式；强化学习输入不与奖励模型在训练阶段遇到样本产生大偏差，维持学习过程稳定性）。
    - 注：KL散度约束二者相似，但并非要保证参数空间的距离保持相似，否则可以直接L2范数来约束，而是要保证两个动作概率的表现相似，因为即使参数相似，输出动作也可能大相径庭。
- PPO优化
    - On-Policy：策略梯度中，负责与环境交互的演员与负责学习的演员相同
    - Off-Policy：两个演员分离，固定一个演员与环境交互而不更新它，将交互得到的轨迹交给另一个负责学习的演员训练。好处是重复利用历史数据，提高效率。PPO属于Off-Policy。依赖Importance Sampling。
- PPO变种
    - PPO-Pentalty：拉格朗日乘数法，把KL散度的限制加入目标函数，变成无约束优化问题
    - PPO-Clip：直接裁剪重要性权重，这样就不需要计算KL散度。Clip函数就是若权重超过1+epsilon则输出1+epsilon，若小于1-epsilon则输出1-epsilon，限制上下界，约束p的差异在合理范围内。

MOSS-RLHF框架
- 影响PPO训练稳定性的7要素：KL惩罚项、奖励值的正则化与裁剪、critic模型的损失裁剪等。
- PPO-max算法，确保RLHF稳定运行（PPO训练中，稳定性和逐渐收敛是困难的）
- Reward Hacking陷入局部最优：增强模型输出与SFT输出空间的KL惩罚力度，确保回复奖励的缓慢稳定提升
- 评估PPO训练成果：因为Reward Hacking所以不能仅依赖回复奖励，需要LLM或人工评估（精心设计prompt如有用性无害性指标）

```py
# 1. Reward Model Training 
# based on LLaMA
import torch 
from transformers.models.llama.modeling_llama import LlamaForCausalLM 

class LlamaRewardModel(LlamaForCausalLM):
    def __init__(self, config, opt, tokenizer):
        super().__init__(config)
        self.opt = opt 
        self.tokenizer = tokenizer
        # add Linear layer reward_head, calculate reward 
        self.reward_head = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, decoder_input, only_last=True):
        attention_mask = decoder_input.ne(self.tokenizer.pad_token_id)
        output = self.model.forward(
            input_ids=decoder_input,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False
        )
        if only_last:
            logits = self.reward_head(output.last_hidden_state[:, -1, :]).squeeze(-1)
        else:
            logits = self.reward_head(output.last_hidden_state).squeeze(-1)
        return (logits,)
    
# loss training, can widen the gap between chose nand rejected response scores on reward model, and can add to final optimization goal for chosen data loss
import torch 

def _criterion(self, model_output, batch, return_output):
    logits, predict_label, *outputs = model_output 
    bs = logits.size(0) // 2 
    preferred_rewards = logits[:bs]
    rejected_rewards = logits[bs:]

    # make preferred labeled data rewards > bad data rewards 
    probs = torch.sigmoid(preferred_rewards - rejected_rewards)
    print(f"self.train_state:{self.train_state}, predict_label: {predict_label}")
    loss = (-torch.log(probs + 1e-5)).mean()

    # language modeling loss
    if self.calculate_lm_loss:
        lm_logits, *_ = outputs 
        scores = lm_logits[:bs, :-1, :]
        preds = scores.argmax(dim=-1)
        label_vec = batch['text_vec'][:bs, 1:].clone()
        loss_mask = batch['loss_mask'][:, 1:]
        label_vec[~loss_mask] = self.tokenizer.null_token_id 
        batch['label_vec'] = label_vec
        lm_loss = super()._criterion((scores, preds), batch, False) # lm loss for chosen only 
        loss = loss + self.lm_loss_factor * lm_loss

    if return_output:
        return (loss, model_output)
    return loss 

# 2. PPO fine tuning 

# Load 4 models: Policy, Critic, Reference, Reward 
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
tokenizer = get_tokenizer(opt)

logging.info(f"Loading policy model from: {opt.policy_model_path}...")
policy_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)
policy_model._set_gradient_checkpointing(policy_model.model, opt.gradient_checkpoint)

logging.info(f"Loading critic model from: {opt.critic_model_path}...")
critic_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)
critic_model._set_gradient_checkpointing(critic_model.model, opt.gradient_checkpoint)

logging.info(f"Loading reference model from: {opt.policy_model_path}...")
ref_model = Llama.from_pretrained(opt.policy_model_path, opt, tokenizer)

logging.info(f"Loading reward model from: {opt.critic_model_path}...")
reward_model = LlamaRewardModel.from_pretrained(opt.critic_model_path, opt, tokenizer)

class RLHFTrainableModelWrapper(nn.Module):
    # wrap policy model and critic model 
    def __init__(self, policy_model, critic_model) -> None:
        super().__init__()
        self.policy_model = policy_model 
        self.critic_model = critic_model

    def forward(self, inputs, **kwargs):
        return self.policy_model(decoder_input=inputs, **kwargs), \
            self.critic_model(decoder_input=inputs, only_last=False, **kwargs)
    
    def train(self, mode=True):
        self.policy_model.train(mode)
        self.critic_model.train(mode)

    def eval(self):
        self.policy_model.eval()
        self.critic_model.eval()

# Experience sampling 
"""
- Read input data, use policy model to respond 
- Reward model to score the response 
- Record response and policy model output probability to replay buffer
"""

@torch.no_grad()
def make_experiences(self):
    # sample from environment 
    start_time = time.time()
    self.model.eval()
    synchronize_if_distributed()
    while len(self.replay_buffer) < self.num_rollouts:
        # get a batch data from generator 
        batch: Dict[str, Any] = next(self.prompt_loader)
        to_cuda(batch)
        context_vec = batch['text_vec'].tolist()

        # get output from policy model 
        _, responses_vec = self.policy_model.generate(batch)
        assert len(context_vec) == len(responses_vec)

        context_vec_sampled, resp_vec_sampled, sampled_vec = self.concat_context_and_response(context_vec, responses_vec)
        sampled_vec = torch.tensor(
            pad_sequences(sampled_vec, pad_value=self.tokenizer.pad_token_id, padding='left'),
            dtype=torch.long, device=self.accelerator.device
        )
        bsz = sampled_vec.size(0)

        rewards, *_ = self.reward_model_forward(sampled_vec)
        rewards = rewards.cpu()
        self.train_metrics.record_metric_many('rewards', rewards.tolist())

        if self.use_reward_scaling:
            # reward scaling 
            rewards_mean, rewards_std = self.running.update(rewards)
            if self.use_reward_norm:
                rewards = (rewards - self.running.mean) / self.running.std 
            else:
                rewards /= self.running.std 
            logging.info(f"Running mean: {self.running.mean}, std: {self.running.std}")
            self.train_metrics.record_metric('reward_mean', rewards_mean)
            self.train_metrics.record_metric('reward_std', rewards_std)
        if self.use_reward_clip:
            # reward clip 
            rewards = torch.clip(rewards, -self.reward_clip, self.reward_clip)

        # calculate log prob and value function beforehand 
        ref_logits, *_ = self.ref_model_forward(sampled_vec)
        logits, *_ = self.policy_model_forward(sampled_vec)
        values, *_ = self.critic_model_forward(sampled_vec)
        torch.cuda.empty_cache()
        assert ref_logits.size(1) == logits.size(1) == values.size(1), \
            f'{ref_logits.size()}, {logits.size()}, {values.size()}'
        ref_logprobs = logprobs_from_logits(ref_logits[:, :-1, :], sampled_vec[:, 1:])
        logprobs = logprobs_from_logits(logits[:, :-1, :], sampled_vec[:, 1:])
        values = values[:, :-1]

        # KL divergence as penalty to ensure RL safe
        kl_pentalty = (-self.kl_penalty_weight * (logprobs - ref_logprobs)).cpu()

        # calculate perplexity during training 
        label = sampled_vec 
        label[label == self.tokenizer.pad_token_id] = self.PAD_TOKEN_LABEL_ID 
        shift_label = label[:, 1:].contiguous()
        valid_length = (shift_label != self.PAD_TOKEN_LABEL_ID).sum(dim=-1)
        shift_logits = logits[..., :-1, :].contiguous()
        ppl_value = self.ppl_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_label.view(-1))
        ppl_value = ppl_value.view(len(logits), -1)
        ppl_value = torch.sum(ppl_value, -1) / valid_length 
        ppl_value = ppl_value.cpu().tolist()

        # calculate policy model original perplexity 
        shift_ref_logits = ref_logits[..., :-1, :].contiguous()
        pp10_value = self.ppl_loss_fct(shift_ref_logits.view(-1, shift_ref_logits.size(-1)), shift_label.view(-1))
        ppl0_value = ppl0_value.view(len(ref_logits), -1)
        ppl0_value = torch.sum(ppl0_value, -1) / valid_length 
        ppl0_value = ppl0_value.cpu().tolist()

        logging.info(f'ppl_value: {ppl_value}')
        logging.info(f'ppl0_value: {ppl0_value}')

        # wrap together: response from sampling, and intermediate variables
        for i in range(bsz):
            resp_length = len(resp_vec_sampled[i])
            penalized_rewards = kl_pentalty[i].clone()
            penalized_rewards[-1] += rewards[i]
            self.train_metrics.record_metric('ref_kl',
                                             (logprobs[i][-resp_length:] - ref_logprobs[i][-resp_length:]).mean().item())
            
            sample = {
                'context_vec': context_vec_sampled[i],
                'context': self.tokenizer.decode(context_vec_sampled[i], skip_special_tokens=False),
                'resp_vec': resp_vec_sampled[i],
                'resp': self.tokenizer.decode(resp_vec_sampled[i], skip_special_tokens=False),
                'reward': penalized_rewards[-resp_length:].tolist(),
                'values': values[i][-resp_length:].tolist(),
                'ref_logprobs': ref_logprobs[i][-resp_length:].tolist(),
                'logprobs': logprobs[i][-resp_length:].tolist(),
                'ppl_value': ppl_value[i],
                'ppl0_value': ppl0_value[i]
            }
            # get pretraining batch data 
            if self.use_ppo_pretrain_loss:
                ppo_batch = next(self.pretrain_loader)
                to_cuda(ppo_batch)
                sample['ppo_context_vec'] = ppo_batch['text_vec'].tolist()
                sample['ppo_loss_mask'] = ppo_batch['loss_mask'].tolist()
            self.replay_buffer.append(sample)
    
    logging.inf(f'Sampled {len(self.replay_buffer)} samples in {(time.time() - start_time):.2f} seconds')
    self.model.train()

# General Advantage Estimation
# based on replay buffer's advantage function and reward function, use data_helper to wrap the estimated values, for policy model and critic model training 

class ExperienceDataset(IterDataset):
    # warp experience data from samples 
    def __init__(self, data, opt, accelerator, mode='train', **kwargs) -> None:
        self.opt = opt 
        self.mode = mode 
        self.accelerator = accelerator
        self.tokenizer = get_tokenizer(opt)
        self.use_ppo_pretrain_loss = opt.use_ppo_pretrain_loss
        self.batch_size = opt.batch_size 
        self.gamma = opt.gamma 
        self.lam = opt.lam 
        self.data = data 
        self.size = len(data)
        if self.accelerator.use_distributed:
            self.size *= self.accelerator.num_processes

    def get_advantages_and_returns(self, rewards: List[float], values: List[float]):
        # GAE algorithm to calculate advantage function and rewards 
        response_length = len(values)
        advantages_reversed = []
        lastgaelam = 0
        for t in reversed(range(response_length)):
            nextvalues = values[t + 1] if t < response_length - 1 else 0.0 
            delta = rewards[t] + self.gamma * nextvalues - values[t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = advantages_reversed[::-1]
        returns = [a + v for a, v in zip(advantages, values)]
        assert len(returns) == len(advantages) == len(values)
        return advantages, returns 
    
    def format(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # format process 
        output = copy.deepcopy(sample)
        advantages, returns = self.get_advantages_and_returns(sample['reward'], sample['values'])
        context_vec, resp_vec = sample['context_vec'], sample['resp_vec']
        assert len(resp_vec) == len(advantages) == len(returns)

        text_vec = context_vec + resp_vec 
        loss_mask = [0] * len(context_vec) + [1] * len(resp_vec)

        output['text'] = self.tokenizer.decode(text_vec, skip_special_tokens=False)
        output['text_vec'] = text_vec 
        output['res_len'] = len(resp_vec)
        output['logprobs'] = [0.] * (len(context_vec) - 1) + output['logprobs']
        output['loss_mask'] = loss_mask 

        output['reward'] = sample['reward']
        output['values'] = [0.] * (len(context_vec) - 1) + output['values']
        output['advantages'] = [0.] * (len(context_vec) - 1) + advantages 
        output['returns'] = [0.] * (len(context_vec) - 1) + returns 
        
        return output 
    
    def batch_generator(self):
        for batch in super().batch_generator():
            yield batch 

    # batch processing for samples 
    def batchify(self, batch_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = {
            'text': [sample['text'] for sample in batch_samples],
            'text_vec': torch.tensor(pad_sequences([sample['text_vec'] for sample in batch_samples],
                                                   pad_value=self.tokenizer.pad_token_id), dtype=torch.long),
            'res_len': [sample['res_len'] for sample in batch_samples],
            'logprobs': torch.tensor(pad_sequences([sample['logprobs'] for sample in batch_samples], pad_value=0.)),
            'loss_mask': torch.tensor(pad_sequences([sample['loss_mask'] for sample in batch_samples], pad_value=0), dtype=torch.bool),
            'ppl_value': torch.tensor([sample['ppl_value'] for sample in batch_samples]),
            'ppl0_value': torch.tensor([sample['ppl_value'] for sample in batch_samples]),
            'reward': [sample['reward'] for sample in batch_samples],
            'values': torch.tensor(pad_sequences([sample['values'] for sample in batch_samples], pad_value=0.)),
            'advantages': torch.tensor(pad_sequences([sample['advantages'] for sample in batch_samples], pad_value=0.)),
            'returns': torch.tensor(pad_sequences([sample['returns'] for sample in batch_samples], pad_value=0.))
        }
        if self.use_ppo_pretrain_loss:
            tmp_ppo_context_vec = []
            for pretrain_data_batch in [sample['ppo_context_vec'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_context_vec.append(one_sample)

            batch['ppo_context_vec'] = torch.tensor(pad_sequences(
                tmp_ppo_context_vec, pad_value=self.tokenizer.pad_token_id
            ), dtype=torch.long)
            del tmp_ppo_context_vec 

            tmp_ppo_loss_mask = []
            for pretrain_data_batch in [sample['ppo_loss_mask'] for sample in batch_samples]:
                for one_sample in pretrain_data_batch:
                    tmp_ppo_loss_mask.append(one_sample)
            batch['ppo_loss_mask'] = torch.tensor(pad_sequences(tmp_ppo_loss_mask, pad_value=0), dtype=torch.bool)
            del tmp_ppo_loss_mask
        return batch 
    
    # Lastly, update policy model and critic model, repeat, use PPO to continue optimize
    def criterion(self, model_output, batch, return_output=False, training=True):
        # optimization goal for policy model and critic model 
        policy_output, critic_output = model_output 
        policy_logits, *_ = policy_output
        values, *_ = critic_output
        values = values[:, :-1]
        loss_mask = batch['loss_mask']
        loss_mask = loss_mask[:, 1:]
        old_values = batch['values']
        old_logprobs = batch['logprobs']
        advantages = batch['advantages']
        returns = batch['returns']
        if self.use_advantage_norm:
            advantages = whiten(advantages, loss_mask, accelerator=self.accelerator)
        if self.use_advantage_clip:
            advantages = torch.clamp(advantages, -self.advantage_clip, self.advantage_clip)
        n = loss_mask.sum()

        logprobs = logprobs_from_logits(policy_logits[:, :-1, :],
                                        batch['text_vec'][:, 1:]) * loss_mask
        # value function loss calculation
        values_clipped = torch.clamp(values,
                                     old_values - self.value_clip,
                                     old_values + self.value_clip,)
        vf_loss1 = (values - returns) ** 2 
        vf_loss2 = (values_clipped - returns) ** 2 

        if self.use_critic_loss_clip:
            vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * loss_mask) / n 
        else:
            vf_loss = 0.5 * torch.sum(vf_loss1 * loss_mask) / n
        vf_clipfrac = torch.sum((vf_loss2 > vf_loss1).float() * loss_mask) / n 

        log_ratio = (logprobs - old_logprobs) * loss_mask 
        ratio = torch.exp(log_ratio)
        with torch.no_grad():
            approx_kl = torch.sum((ratio - 1) - log_ratio) / n 

        pg_loss1 = -advantages * ratio 
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.pg_clip, 1.0 + self.pg_clip)

        # policy model loss clip
        if self.use_policy_loss_clip:
            pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * loss_mask) / n 
        else:
            pg_loss = torch.sum(pg_loss1 * loss_mask) / n 
        pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * loss_mask) / n 

        # entropy regularization
        if self.use_entropy_loss:
            ent = get_category_distribution_entropy(len(policy_logits), policy_logits[:, :-1, :])
            entro_loss = torch.abs(torch.sum(ent * loss_mask) / n - self.entropy_clip)

        # pretraining loss calculation 
        if self.use_ppo_pretrain_loss:
            pretrain_sampled_vec = batch['ppo_context_vec']
            scroes, *_ = self.policy_model_forward(pretrain_sampled_vec)
            scores = scores[:, :-1, :]
            preds = scores.argmax(dim=-1)

            ppo_label_vec = batch['ppo_context_vec'][:, 1:].clone()
            ppo_loss_mask = batch['ppo_loss_mask'][:, 1:]
            ppo_label_vec[~ppo_loss_mask] = self.tokenizer.pad_token_id 

            labels: torch.LongTensor = ppo_label_vec

            score_view = scores.reshape(-1, scores.size(-1)) # bs * num_tokens, vocab_size 
            pretrain_loss = self.loss_fn(score_view, labels.reshape(-1)).sum()

            # token prediction precision
            notnull = labels.ne(self.tokenizer.pad_token_id)
            target_tokens = notnull.sum()
            correct = ((labels == preds) * notnull).sum()

            # avg loss 
            pretrain_loss = pretrain_loss / target_tokens
            if self.use_entropy_loss:
                loss1 = pg_loss + self.vf_loss_weight * vf_loss + self.entropy_loss_weight * entro_loss 
            else:
                loss1 = pg_loss + self.vf_loss_weight * vf_loss 
            loss2 = self.ppo_pretrain_loss_weight * pretrain_loss 
            loss = loss1 + loss2 
        else:
            if self.use_entropy_loss:
                loss = pg_loss + self.vf_loss_weight * vf_loss + self.entropy_loss_weight * entro_loss 
            else:
                loss = pg_loss + self.vf_loss_weight * vf_loss 
        if self.use_ppo_pretrain_loss:
            if return_output:
                return loss1, loss2, model_output 
            else:
                return loss1, loss2 
        if return output:
            return loss, model_output 
        
        return loss 
```

<!-- TOC --><a name="7-applications"></a>
## 7. Applications 

<!-- TOC --><a name="8-evaluations"></a>
## 8. Evaluations







