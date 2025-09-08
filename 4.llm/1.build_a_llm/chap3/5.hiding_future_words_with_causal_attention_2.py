
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

    # 2.利用dropout掩码额外的注意力权重
    text = "Your journey starts with one step"
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3
    d_out = 2               # 输出嵌入维度 d_out=2    在类GPT模型中,输入和输出的维度相同 此处不同是便于理解

    sa_v2 = SelfAttention_v2(d_in, d_out)
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    attn_weights = t.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
    print(attn_weights)
    """
    tensor([[0.1717, 0.1762, 0.1761, 0.1555, 0.1627, 0.1579],
            [0.1636, 0.1749, 0.1746, 0.1612, 0.1605, 0.1652],
            [0.1637, 0.1749, 0.1746, 0.1611, 0.1606, 0.1651],
            [0.1636, 0.1704, 0.1702, 0.1652, 0.1632, 0.1674],
            [0.1667, 0.1722, 0.1721, 0.1618, 0.1633, 0.1639],
            [0.1624, 0.1709, 0.1706, 0.1654, 0.1625, 0.1682]],
           grad_fn=<SoftmaxBackward0>)
    """

    context_length = attn_scores.shape[0]
    mask_simple = t.tril(t.ones(context_length, context_length))
    print(mask_simple)
    """
    tensor([[1., 0., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0., 0.],
            [1., 1., 1., 0., 0., 0.],
            [1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1., 1.]])      tril() 创建一个对角线以上元素为0的掩码
    """
    masked_simple = attn_weights * mask_simple      # 哈达玛积:*(逐元素相乘)
    print(masked_simple)
    """
    tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
            [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
            [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
            [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
           grad_fn=<MulBackward0>)
    """

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
    """

    # 对注意力权重进行dropout
    t.manual_seed(123)
    dropout = t.nn.Dropout(0.5)     # 50%的dropout率  掩码一半的注意力权重  后面章节训练GPT时,将使用较低的dropout,eg:10% 20%
    example = t.ones(6, 6)
    output = dropout(example)       # 矩阵有一半元素被随机置0,为补偿减少的活跃元素,矩阵中剩余的元素按1/0.5=2的比例进行放大
    print(output)                   # 这种放大对维持注意力权重的整体平衡很重要,确保在训练和推理过程中,注意力机制的平均影响保持一致
    """
    tensor([[2., 2., 2., 2., 2., 2.],
            [0., 2., 0., 0., 0., 0.],
            [0., 0., 2., 0., 2., 0.],
            [2., 2., 0., 0., 0., 2.],
            [2., 0., 0., 0., 0., 2.],
            [0., 2., 0., 0., 0., 0.]])
    """

    do_attn_weights = dropout(attn_weights)
    print(do_attn_weights)
    """
    tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.6206, 0.0000, 0.0000, 0.0000],
            [0.0000, 0.4921, 0.0000, 0.4638, 0.0000, 0.0000],
            [0.0000, 0.3966, 0.3968, 0.3775, 0.3941, 0.0000],
            [0.3869, 0.3327, 0.0000, 0.0000, 0.3331, 0.3058]], grad_fn=<MulBackward0>)  因为操作系统的差异,此处输出可能会不同
    """
