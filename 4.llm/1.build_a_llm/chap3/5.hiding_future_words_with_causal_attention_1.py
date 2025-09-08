
import torch as t
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # 使用nn.Linear进行优化
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)      # 因为采用优化的权重初始化方案,有助模型训练的稳定性和有效性
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = t.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vector = attn_weights @ values
        return context_vector

"""
    利用因果注意力隐藏未来词汇
        1.因果注意力的掩码实现
        2.利用dropout掩码额外的注意力权重   有效减少过拟合
        3.实现一个简化的因果注意力类
"""
if __name__ == '__main__':
    t.manual_seed(789)

    # 1.因果注意力的掩码实现
    text = "Your journey starts with one step"
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3
    d_out = 2               # 输出嵌入维度 d_out=2    在类GPT模型中,输入和输出的维度相同 此处不同是便于理解

    # a:计算注意力权重
    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = t.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights,attn_weights.shape)
    """
    tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
            [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
            [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
            [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
            [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
            [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]], grad_fn=<SoftmaxBackward0>)  torch.Size([6, 6])
    """

    # b:计算掩码矩阵
    context_length = attn_scores.shape[0]
    mask_simple = t.tril(t.ones(context_length, context_length))
    print(mask_simple)
    """
    tensor([[1., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1., 1.]])      tril() 创建一个对角线以上元素为0的掩码 size(6*6)
    """

    # c:计算掩码注意力权重
    masked_simple = attn_weights * mask_simple      # 将注意力矩阵与掩码矩阵相乘,使对角线上方的值变为0
    print(masked_simple)                            # 哈达玛积:*(逐元素相乘)
    """
    tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
            [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
            [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
            [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]], grad_fn=<MulBackward0>)  
    """

    # d1:归一化掩码注意力权重
    row_sums = masked_simple.sum(dim=-1, keepdim=True)
    masked_simple_norm = masked_simple / row_sums
    print(masked_simple_norm)
    """
    tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
            [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
            [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
            [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
           grad_fn=<DivBackward0>)
    """

    # d2:归一化掩码注意力权重 softmax
    mask = t.triu(t.ones(context_length, context_length), diagonal=1)
    masked = attn_scores.masked_fill(mask.bool(), -t.inf)
    print(masked)
    """
    tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
            [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
            [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
            [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
            [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
            [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
           grad_fn=<MaskedFillBackward0>)
    """

    attn_weights = t.softmax(masked / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    """
    tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
            [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
            [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
            [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
           grad_fn=<SoftmaxBackward0>)
    注意:
        掩码和重新归一化后,是在一个较小的子集重新计算softmax分数,因为被掩码的位置不参与softmax计算 
        掩码和重新归一化后,被掩码的位置不会以任何实际的方式影响softmax分数
        即注意力权重的分布就像最初仅在未掩码的位置计算一张,保证不会有来自未来或其他被掩码的词元的信息泄露
        
    softmax:
        当输入中出现无穷大值时,softmax将这些值视为零概率 exp(-∞)≈0 无限接近于0
        创建对角线以上全是1的掩码,将1替换未-inf(负无穷大),实现更高效的掩码
    """

    # 现在可以使用修改后的注意力权重计算上下文向量
    values = sa_v2.W_value(inputs)
    context_vector = attn_weights @ values
    print(context_vector,context_vector.shape)
    """
    tensor([[-0.0872,  0.0286],
            [-0.0991,  0.0501],
            [-0.0999,  0.0633],
            [-0.0983,  0.0489],
            [-0.0514,  0.1098],
            [-0.0754,  0.0693]], grad_fn=<MmBackward0>) torch.Size([6, 2])
    """

