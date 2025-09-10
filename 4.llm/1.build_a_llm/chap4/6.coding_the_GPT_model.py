
import tiktoken
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

# demo4.2
def get_config(base_config, model_name="gpt2-small"):
    GPT_CONFIG = base_config.copy()

    if model_name == "gpt2-small":
        GPT_CONFIG["embedding_dim"] = 768
        GPT_CONFIG["n_layers"] = 12
        GPT_CONFIG["n_heads"] = 12

    elif model_name == "gpt2-medium":
        GPT_CONFIG["embedding_dim"] = 1024
        GPT_CONFIG["n_layers"] = 24
        GPT_CONFIG["n_heads"] = 16

    elif model_name == "gpt2-large":
        GPT_CONFIG["embedding_dim"] = 1280
        GPT_CONFIG["n_layers"] = 36
        GPT_CONFIG["n_heads"] = 20

    elif model_name == "gpt2-xl":
        GPT_CONFIG["embedding_dim"] = 1600
        GPT_CONFIG["n_layers"] = 48
        GPT_CONFIG["n_heads"] = 25

    else:
        raise ValueError(f"Incorrect model name {model_name}")
    return GPT_CONFIG

def calculate_size(model):  # based on chapter code
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")

    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")

"""
    实现GPT模型
        DummyGPTModel 
            DummyTransformerBlock ---> TransformerBlock
            DummyLayerNorm ---> LayerNorm
        
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # a:实现GPTModel
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # 词汇表大小     BPE分词器使用的由50257个单词组成的词汇表
        "context_length": 1024,     # 上下文长度     模型通过位置嵌入能够处理的最大输入词元数量
        "embedding_dim": 768,       # 嵌入维度       将每个词元转为768维的向量
        "n_heads": 12,              # 注意力头的数量  多头注意力机制中注意力头的数量
        "n_layers": 12,             # 层数          模型中的Transformer块数量
        "drop_rate": 0.1,           # dropout率     dropout机制的强制 0.1表示有10%的隐藏单元被随机丢弃,防止过拟合
        "qkv_bias": False           # 查询-键-值偏置  是否在多头注意力机制的线性层中添加一个偏置向量,用于查询,键和值的计算 刚开始禁用
    }

    tokenizer = tiktoken.get_encoding("gpt2")       # tiktoken分词器对批输入进行分词处理
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    batch.append(t.tensor(tokenizer.encode(text1)))
    batch.append(t.tensor(tokenizer.encode(text2)))
    batch = t.stack(batch, dim=0)

    model = GPTModel(GPT_CONFIG_124M)
    out = model(batch)
    print("Input batch:", batch)
    print("Output shape:", out.shape)
    print(out)
    """
    Input batch: 
    tensor([[6109, 3626, 6100,  345],
            [6109, 1110, 6622,  257]])
    Output shape: torch.Size([2, 4, 50257])
    tensor([[[ 0.1381,  0.0077, -0.1963,  ..., -0.0222, -0.1060,  0.1717],
             [ 0.3865, -0.8408, -0.6564,  ..., -0.5163,  0.2369, -0.3357],
             [ 0.6989, -0.1829, -0.1631,  ...,  0.1472, -0.6504, -0.0056],
             [-0.4290,  0.1669, -0.1258,  ...,  1.1579,  0.5303, -0.5549]],
    
            [[ 0.1094, -0.2894, -0.1467,  ..., -0.0557,  0.2911, -0.2824],
             [ 0.0882, -0.3552, -0.3527,  ...,  1.2930,  0.0053,  0.1898],
             [ 0.6091,  0.4702, -0.4094,  ...,  0.7688,  0.3787, -0.1974],
             [-0.0612, -0.0737,  0.4751,  ...,  1.2463, -0.3834,  0.0609]]], grad_fn=<UnsafeViewBackward0>)
    """

    # b:权重共享
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")
    """
    Total number of parameters: 163,009,536
    """

    print("Token embedding layer shape:", model.token_embedding.weight.shape)
    print("Output layer shape:", model.out_head.weight.shape)
    """
    Token embedding layer shape: torch.Size([50257, 768])
    Output layer shape: torch.Size([50257, 768])
    """

    total_params_gpt2 = total_params - sum(p.numel() for p in model.out_head.parameters())
    print(f"Number of trainable parameters considering weight tying: {total_params_gpt2:,}")
    """
    Number of trainable parameters considering weight tying: 124,412,160
    """

    # c:GPTModel对象中1.63亿个参数的内存需求
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")
    """
    Total size of the model: 621.83 MB
    """

    # demo4.1: 前馈模块和注意力模块的参数量,并进行对比
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "embedding_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }

    block = TransformerBlock(GPT_CONFIG_124M)
    print(block)
    """
    TransformerBlock(
      (attention): MultiHeadAttention(
        (W_query): Linear(in_features=768, out_features=768, bias=False)
        (W_key): Linear(in_features=768, out_features=768, bias=False)
        (W_value): Linear(in_features=768, out_features=768, bias=False)
        (output_projection): Linear(in_features=768, out_features=768, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (feedforward): FeedForward(
        (layers): Sequential(
          (0): Linear(in_features=768, out_features=3072, bias=True)
          (1): GELU()
          (2): Linear(in_features=3072, out_features=768, bias=True)
        )
      )
      (norm1): LayerNorm()
      (norm2): LayerNorm()
      (drop_shortcut): Dropout(p=0.1, inplace=False)
    )    
    """

    total_params = sum(p.numel() for p in block.feedforward.parameters())
    print(f"Total number of parameters in feed forward module: {total_params:,}")
    """
    Total number of parameters in feed forward module: 4,722,432
    """

    total_params = sum(p.numel() for p in block.attention.parameters())
    print(f"Total number of parameters in attention module: {total_params:,}")
    """
    Total number of parameters in attention module: 2,360,064
    """

    # demo4.2:初始化更大GPT模型
    for model_abbrev in ("small", "medium", "large", "xl"):
        model_name = f"gpt2-{model_abbrev}"
        CONFIG = get_config(GPT_CONFIG_124M, model_name=model_name)
        model = GPTModel(CONFIG)
        print(f"{model_name}:")
        calculate_size(model)
    """
    gpt2-small:
        Total number of parameters: 163,009,536
        Number of trainable parameters considering weight tying: 124,412,160
        Total size of the model: 621.83 MB
    gpt2-medium:
        Total number of parameters: 406,212,608
        Number of trainable parameters considering weight tying: 354,749,440
        Total size of the model: 1549.58 MB
    gpt2-large:
        Total number of parameters: 838,220,800
        Number of trainable parameters considering weight tying: 773,891,840
        Total size of the model: 3197.56 MB
    gpt2-xl:
        Total number of parameters: 1,637,792,000
        Number of trainable parameters considering weight tying: 1,557,380,800
        Total size of the model: 6247.68 MB    
    """