
import torch as t

def softmax_naive(x):
    return t.exp(x) / t.exp(x).sum(dim=0)

"""
    无可训练权重的简单自注意力机制
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 将输入序列按chap2的方式嵌入为三维向量
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    # 1.计算输入2的注意力权重和上下文向量
    # a.计算注意力分数 ω
    query = inputs[1]                           # 查询词:journey
    attn_scores_2 = t.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = t.dot(x_i,query)     # 注意力分数=查询词·序列中元素
    print(attn_scores_2)
    """
    tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])
    """

    # b.注意力分数归一化(注意力权重归一化)  获取总和为1的注意力权重
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())
    """
    Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
    Sum: tensor(1.0000)
    """

    # softmax归一化 (保证注意力权重总是正值,方便将输出解释为概率,权重越高则重要程度越高 实际使用中更常见)
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())
    """
    Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    Sum: tensor(1.)
    """

    # torch.softmax 防溢出
    attn_weights_2 = t.softmax(attn_scores_2, dim=0)
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())
    """
    Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    Sum: tensor(1.)
    """

    # c.计算上下文向量
    query = inputs[1]           # 查询词:journey
    context_vector_2 = t.zeros(query.shape)
    for i, x_i in enumerate(inputs):
        context_vector_2 += attn_weights_2[i] * x_i
    print(context_vector_2)
    """
    tensor([0.4419, 0.6515, 0.5683])
    """

    # 2.计算所有输入的注意力去找你和上下文向量
    attn_scores = t.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = t.dot(x_i, x_j)
    print(attn_scores)
    """
    tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
            [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
            [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
            [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
            [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
            [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])  每个元素表示每对输入之间的注意力分数
    """

    attn_scores = inputs @ inputs.T
    print(attn_scores)
    """
    tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
            [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
            [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
            [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
            [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
            [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])  for循环太慢,改为矩阵乘法  矩阵乘法
    """

    attn_weights = t.softmax(attn_scores, dim=-1)
    print(attn_weights)
    """
    tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
            [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
            [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
            [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
            [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
            [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])  每行进行归一化,每一行的值总和为1
    注:
        dim=-1,表示在张量的最后一个维度上进行归一化 
        eg:二维张量[行,列],dim=-1表示在列进行归一化即每行的总和为1
    """

    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))
    """
    Row 2 sum: 1.0
    All row sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])
    """

    all_context_vectorss = attn_weights @ inputs
    print(all_context_vectorss)
    print("Previous 2nd context vector:", context_vector_2)
    """
    tensor([[0.4421, 0.5931, 0.5790],
            [0.4419, 0.6515, 0.5683],
            [0.4431, 0.6496, 0.5671],
            [0.4304, 0.6298, 0.5510],
            [0.4671, 0.5910, 0.5266],
            [0.4177, 0.6503, 0.5645]])
    Previous 2nd context vector: tensor([0.4419, 0.6515, 0.5683])
    """