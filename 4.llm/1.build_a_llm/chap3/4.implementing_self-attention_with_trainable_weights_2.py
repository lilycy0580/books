
import torch as t
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(t.rand(d_in, d_out))
        self.W_key = nn.Parameter(t.rand(d_in, d_out))
        self.W_value = nn.Parameter(t.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T  # omega  ω                                  # 注意力分数
        attn_weights = t.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)       # 注意力权重

        context_vector = attn_weights @ values                                      # 上下文向量
        return context_vector


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
    带可训练权重的自注意力机制
        1.计算第2个输入的上下文向量 z(2)
        2.计算第所有输入的上下文向量 z
"""
if __name__ == '__main__':
    t.manual_seed(123)

    text = "Your journey starts with one step"
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    # 1.计算所有输入的上下文向量 z
    d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3
    d_out = 2               # 输出嵌入维度 d_out=2    在类GPT模型中,输入和输出的维度相同 此处不同是便于理解

    sa_v1 = SelfAttention_v1(d_in, d_out)
    context_vector_v1 = sa_v1(inputs)
    print(context_vector_v1)
    """
    tensor([[0.2996, 0.8053],
            [0.3061, 0.8210],
            [0.3058, 0.8203],
            [0.2948, 0.7939],
            [0.2927, 0.7891],
            [0.2990, 0.8040]], grad_fn=<MmBackward0>)   第2行与上一节代码的输出一致
    """

    t.manual_seed(789)
    sa_v2 = SelfAttention_v2(d_in, d_out)
    context_vector_v2 = sa_v2(inputs)
    print(context_vector_v2)
    """
    tensor([[-0.0739,  0.0713],
            [-0.0748,  0.0703],
            [-0.0749,  0.0702],
            [-0.0760,  0.0685],
            [-0.0763,  0.0679],
            [-0.0754,  0.0693]], grad_fn=<MmBackward0>)     
    
    SelfAttention_v1与SelfAttention_v2因为使用不同的初始值权重矩阵给出不同的输出,由nn.Linear使用了一个更复杂的权值初始化导致
    """

    # demo:将SelfAttention_v2的权重转移到SelfAttention_v1中,查看输出结果
    sa_v1.W_query = t.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = t.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = t.nn.Parameter(sa_v2.W_value.weight.T)
    context_vector_v1 = sa_v1(inputs)
    print(context_vector_v1,t.equal(context_vector_v1, context_vector_v2))
    """
    tensor([[-0.0739,  0.0713],
            [-0.0748,  0.0703],
            [-0.0749,  0.0702],
            [-0.0760,  0.0685],
            [-0.0763,  0.0679],
            [-0.0754,  0.0693]], grad_fn=<MmBackward0>) True
    """



