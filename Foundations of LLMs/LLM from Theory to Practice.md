LLM from Theory to Practice, by Qi Zhang, 2024

<img src="https://github.com/user-attachments/assets/58741520-5516-4c3c-8a65-26f2e158a5d9" width="32%" height="32%">

## 2. LLM Intro

- 

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
- RoPE

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

## 3.




