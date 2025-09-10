
import torch as t
import torch.nn as nn

# 掩码多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 整除 将最终输出的总维度d_out平均分配给所有的注意力头,计算出每个头应该负责的维度大小

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.output_projection = nn.Linear(d_out, d_out)  # Transformer的多头注意力机制中，该层称为输出投影层
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

# 5层的神经网络 快捷连接
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:     # self.use_shortcut=True,添加快捷连接
                x = x + layer_output
            else:
                x = layer_output
        return x

# Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = MultiHeadAttention(
            d_in=cfg["embedding_dim"],
            d_out=cfg["embedding_dim"],
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

"""
    连接Transformer块中的注意力层和线性层
"""
if __name__ == '__main__':
    t.manual_seed(123)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # 词汇表大小     BPE分词器使用的由50257个单词组成的词汇表
        "context_length": 1024,     # 上下文长度     模型通过位置嵌入能够处理的最大输入词元数量
        "embedding_dim": 768,             # 嵌入维度       将每个词元转为768维的向量
        "n_heads": 12,              # 注意力头的数量  多头注意力机制中注意力头的数量
        "n_layers": 12,             # 层数          模型中的Transformer块数量
        "drop_rate": 0.1,           # dropout率     dropout机制的强制 0.1表示有10%的隐藏单元被随机丢弃,防止过拟合
        "qkv_bias": False           # 查询-键-值偏置  是否在多头注意力机制的线性层中添加一个偏置向量,用于查询,键和值的计算 刚开始禁用
    }

    x = t.rand(2, 4, 768)           # [batch_size, num_tokens, emb_dim]
    block = TransformerBlock(GPT_CONFIG_124M)
    output = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    """
    Input shape: torch.Size([2, 4, 768])
    Output shape: torch.Size([2, 4, 768])
    """

    """
    TransformerBlock块:
        多头注意力机制
        前馈神经网络
        
        层归一化应用于这两个组件之前
            前层归一化   自注意力和前馈神经网络之前应用层归一化
            后层归一化   自注意力和前馈神经网络之后应用层归一化(早期的Transformer模型,导致较差的训练结果)
            
        dropout应用于这两个组件之后,方便对模型进行正则化并防止过拟合   

        实现前向传播,每个组件后跟着一个快捷连接,将块的输入加入到输出上,有注于在训练过程中使梯度在网络中流动,改善深度模型的学习效果
        
    Transformer块架构在处理数据序列时不会改变它们在网络中的状态
        输入序列的物理维度在通过Transformer块时不改变,但是每个输出向量中的内容都要重新编码,整合来自整个输入序列的上下文信息
    """



























