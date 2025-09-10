import torch as t
import torch.nn as nn

# 掩码多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (dim_out % num_heads == 0), "dim_out must be divisible by num_heads"

        self.d_out = dim_out
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads  # 整除 将最终输出的总维度d_out平均分配给所有的注意力头,计算出每个头应该负责的维度大小

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.output_projection = nn.Linear(dim_out, dim_out)  # Transformer的多头注意力机制中，该层称为输出投影层
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask",t.triu(t.ones(context_length, context_length),diagonal=1))

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape

        keys = self.W_key(x)        # Shape  (batch_size, num_tokens, dim_out)
        queries = self.W_query(x)   # 分割矩阵 (batch_size, num_tokens, num_heads, head_dim)
        values = self.W_value(x)    # 矩阵转置 (batch_size, num_heads, num_tokens, head_dim)

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)     # 注意此处values已经转置,后续使用时需再次转置

        # (batch_size, num_heads, num_tokens, head_dim)-->(batch_size, num_heads, head_dim, num_tokens)
        attn_scores = queries @ keys.transpose(2, 3)  # 转置 (b,n,d)-->(b,d,n)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -t.inf)

        attn_weights = t.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # (batch_size, num_heads, num_tokens, head_dim)-->(b, num_tokens, num_heads, head_dim)
        context_vectors = (attn_weights @ values).transpose(1, 2)

        # self.d_out = self.num_heads * self.head_dim
        context_vectors = context_vectors.contiguous().view(batch_size, num_tokens, self.d_out)
        context_vectors = self.output_projection(context_vectors)  # 已实例化的神经网络层,需输入数据才能执行前向传播
        return context_vectors

# 前馈网络
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + t.tanh(t.sqrt(t.tensor(2.0 / t.pi)) *  (x + 0.044715 * t.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(                                    # 维度先升,激活后,再降,允许模型探索丰富的表示空间
            nn.Linear(cfg["embedding_dim"], 4 * cfg["embedding_dim"]),  # 768,768*4
            GELU(),
            nn.Linear(4 * cfg["embedding_dim"], cfg["embedding_dim"]))  # 768*4,768

    def forward(self, x):
        return self.layers(x)

# 层归一化
class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):                      # embedding_dim:词嵌入的维度 对应输入张量x的最后一个维度
        super().__init__()
        self.eps = 1e-5                                     # 防止除以零的数值不稳定情况
        self.scale = nn.Parameter(t.ones(embedding_dim))    # 可学习的缩放参数 γ
        self.shift = nn.Parameter(t.zeros(embedding_dim))   # 可学习的平移参数 β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # unbiased=False 嵌入维度n比较大的大语言模型,使用n和n-1的差异可以忽略
        norm_x = (x - mean) / t.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    """
    unbiased=True
        贝塞尔修正:样本方差的估计中使用n-1作为分母,调整偏差
    """

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            dim_in=cfg["embedding_dim"],
            dim_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.feedforward = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Shortcut connection for attention block
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut        # Add the original input back

        # Shortcut connection for feed forward block
        shortcut = x
        x = self.norm2(x)
        x = self.feedforward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut        # Add the original input back
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["embedding_dim"])

        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)
        pos_embeddings = self.pos_embedding(t.arange(seq_len, device=in_idx.device))
        x = token_embeddings + pos_embeddings     # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_embedding(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits