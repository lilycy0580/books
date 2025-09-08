
import torch as t
import torch.nn as nn

# 一个高效的多头注意力 从一个多头注意力层开始,在内部将这个层分割成单独的注意力头
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

"""
    将单头注意力扩展到多头注意力
        1.叠加多个单头注意力层
        2.通过权重划分实现多头注意力
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # 2.通过权重划分实现多头注意力
    text = "Your journey starts with one step"
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    batch = t.stack((inputs, inputs), dim=0)    # 按batch输入
    batch_size, context_length, d_in = batch.shape
    d_out = 2
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vectors = mha(batch)
    print(context_vectors,context_vectors.shape)
    """
    tensor([[[0.3190, 0.4858],
             [0.2943, 0.3897],
             [0.2856, 0.3593],
             [0.2693, 0.3873],
             [0.2639, 0.3928],
             [0.2575, 0.4028]],
    
            [[0.3190, 0.4858],
             [0.2943, 0.3897],
             [0.2856, 0.3593],
             [0.2693, 0.3873],
             [0.2639, 0.3928],
             [0.2575, 0.4028]]], grad_fn=<ViewBackward0>) torch.Size([2, 6, 2])
    torch.Size([2, 6, 2])   
        2个输入文本 每个输入中有6个词元 每个词元的2维嵌入
    """

    a = t.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],
                    [0.8993, 0.0390, 0.9268, 0.7388],
                    [0.7179, 0.7058, 0.9156, 0.4340]],
                   [[0.0772, 0.3565, 0.1479, 0.5331],
                    [0.4066, 0.2318, 0.4545, 0.9737],
                    [0.4606, 0.5159, 0.4220, 0.5786]]]])
    product = a @ a.transpose(2, 3)
    print(product)
    """
    tensor([[[[1.3208, 1.1631, 1.2879],
              [1.1631, 2.2150, 1.8424],
              [1.2879, 1.8424, 2.0402]],
             [[0.4391, 0.7003, 0.5903],
              [0.7003, 1.3737, 1.0620],
              [0.5903, 1.0620, 0.9912]]]])
    """

    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T
    print("First head:", first_res)
    """
    First head: 
    tensor([[1.3208, 1.1631, 1.2879],
            [1.1631, 2.2150, 1.8424],
            [1.2879, 1.8424, 2.0402]])
    """

    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print("Second head:", second_res)
    """
    Second head: 
    tensor([[0.4391, 0.7003, 0.5903],
            [0.7003, 1.3737, 1.0620],
            [0.5903, 1.0620, 0.9912]])
    """

    # demo:初始化GPT-2大小的注意力模块
    # 最小的GPT-2模型 参数量1.17亿 12个注意力头 上下文向量嵌入维度为768 支持1024个词元的上下文长度
    d_in, context_length,d_out = 768, 1024,768
    mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=12)