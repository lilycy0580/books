
import torch as t

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

    # 1.计算第2个输入的上下文向量 z(2)
    x_2 = inputs[1]         # 第2个输入元素
    d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3
    d_out = 2               # 输出嵌入维度 d_out=2    在类GPT模型中,输入和输出的维度相同 此处不同是便于理解

    # a:初始化三个权重,requires_grad=False减少输出中的其他项 若需要在训练模型中使用这些权重矩阵,需设为True,方便在训练中更新这些权重
    W_query = t.nn.Parameter(t.rand(d_in, d_out), requires_grad=False)
    W_key = t.nn.Parameter(t.rand(d_in, d_out), requires_grad=False)
    W_value = t.nn.Parameter(t.rand(d_in, d_out), requires_grad=False)

    # b1:计算查询向量,键向量和值向量    矩阵乘法:@(1*3 3*2 = 1*2)
    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2,key_2,value_2)
    """
    tensor([0.4306, 1.4551])    query_2
    tensor([0.4433, 1.1419])    key_2
    tensor([0.3951, 1.0037])    value_2
    """

    # b2.获取所有输入序列的键向量,值向量   只计算一个上下文向量z(2)
    keys = inputs @ W_key
    values = inputs @ W_value
    print("keys.shape:", keys.shape)
    print("values.shape:", values.shape)
    """
    keys.shape: torch.Size([6, 2])
    values.shape: torch.Size([6, 2])    将6个输入词元从三维空间映射到二维嵌入空间
    """

    # c.计算注意力分数 ω_22
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print(query_2,query_2.shape)
    print(keys_2,keys_2.shape)
    print(attn_score_22)
    """
    tensor([0.4306, 1.4551]) torch.Size([2])
    tensor([0.4433, 1.1419]) torch.Size([2])
    tensor(1.8524)
    """

    attn_scores_2 = query_2 @ keys.T
    print(query_2,query_2.shape)
    print(keys.T,keys.T.shape)
    print(attn_scores_2)
    """
    tensor([0.4306, 1.4551]) torch.Size([2])
    tensor([[0.3669, 0.4433, 0.4361, 0.2408, 0.1827, 0.3275],
            [0.7646, 1.1419, 1.1156, 0.6706, 0.3292, 0.9642]]) torch.Size([2, 6])
    tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])    计算所有输入的注意力分数
    """

    # d:将注意力分数转换为注意力权重
    dim_keys = keys.shape[1]
    attn_weights_2 = t.softmax(attn_scores_2 / dim_keys ** 0.5, dim=-1)
    print(attn_weights_2)
    """
    tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
    缩放注意力分数并使用softmax进行归一化计算注意力权重
    缩放方式:
        将注意力分数/键向量的嵌入维度的平方根   dim_keys=2 keys.size=[6,2]
    """

    # e:计算上下文向量 值向量进行加权求和计算上下文向量,注意力权重作为加权因子,衡量每个值向量的重要性
    context_vector_2 = attn_weights_2 @ values
    print(context_vector_2)
    """
    tensor([0.3061, 0.8210])
    """






