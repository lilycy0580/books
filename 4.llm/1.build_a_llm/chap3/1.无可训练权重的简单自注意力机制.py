import torch
import tiktoken

# 有溢出风险
def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)


if __name__ == '__main__':
    torch.manual_seed(123)

    # 输入文本：Your journey starts with one step
    # TODO 查询词:journey 与其他词元的注意力分数
    # 1.获取嵌入向量
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],    # Your     (x^1)
         [0.55, 0.87, 0.66],    # journey  (x^2)
         [0.57, 0.85, 0.64],    # starts   (x^3)
         [0.22, 0.58, 0.33],    # with     (x^4)
         [0.77, 0.25, 0.10],    # one      (x^5)
         [0.05, 0.80, 0.55]]    # step     (x^6)
    )

    # 2.计算注意力分数  查询词与各个词元的点积
    query = inputs[1]           # 2nd input token is the query
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i,query)  # dot product (transpose not necessary here since they are 1-dim vectors)
    print(attn_scores_2)        # tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

    # 注意力分数归一化  方式三种,取pytorch的softmax版
    attn_weights_tmp1 = attn_scores_2 / attn_scores_2.sum()
    print("Attention weights:", attn_weights_tmp1)
    print("Sum:", attn_weights_tmp1.sum())

    attn_weights_tmp2 = softmax_naive(attn_scores_2)
    print("Attention weights:", attn_weights_tmp2)
    print("Sum:", attn_weights_tmp2.sum())

    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)  # pytorch版本的softmax版本
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # 3.计算上下文向量
    context_vec_2 = torch.zeros(query.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i
    print(context_vec_2)        # tensor([0.4419, 0.6515, 0.5683])

    # TODO 计算所有输入词元的注意力权重
    # 1.注意力权重及其归一化
    # 方式一: v 慢
    attn_scores = torch.empty(6, 6)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores[i, j] = torch.dot(x_i, x_j)
    print(attn_scores)

    attn_weights = torch.softmax(attn_scores, dim=-1)
    print(attn_weights)

    # 方式二: v 快 哈达玛积
    attn_scores = inputs @ inputs.T
    print(attn_scores)
    attn_weights = torch.softmax(attn_scores, dim=-1)   # dim=-1,表示最后一个维度进行归一化  二维[行,列],对列进行归一化,每行总和=1
    print(attn_weights)

    row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    print("Row 2 sum:", row_2_sum)
    print("All row sums:", attn_weights.sum(dim=-1))

    all_context_vecs = attn_weights @ inputs
    print(all_context_vecs)
    print("Previous 2nd context vector:\n", context_vec_2)
    """
    tensor([[0.4421, 0.5931, 0.5790],
            [0.4419, 0.6515, 0.5683],
            [0.4431, 0.6496, 0.5671],
            [0.4304, 0.6298, 0.5510],
            [0.4671, 0.5910, 0.5266],
            [0.4177, 0.6503, 0.5645]])
            
    Previous 2nd context vector: 
        tensor([0.4419, 0.6515, 0.5683])
    """