import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):                              # 该类继承nn.Module类,需实现init与forward两个方法

    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))    # 初始化三种权重,q,k,v
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key                                   # x为嵌入向量,计算三种向量,q,k,v  矩阵乘法
        queries = x @ self.W_query
        values = x @ self.W_value

        attn_scores = queries @ keys.T                          # 注意力分数转换为注意力权重,注意归一化,避免梯度过小
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values                     # 计算上下文向量
        return context_vec


class SelfAttention_v2(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)    # 使用nn.Linear代替nn.Parameter,方便快速执行矩阵乘法
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attn_weights @ values
        return context_vec


if __name__ == '__main__':
    torch.manual_seed(123)

    # 嵌入向量
    inputs = torch.tensor(
        [[0.43, 0.15, 0.89],    # Your     (x^1)
         [0.55, 0.87, 0.66],    # journey  (x^2)
         [0.57, 0.85, 0.64],    # starts   (x^3)
         [0.22, 0.58, 0.33],    # with     (x^4)
         [0.77, 0.25, 0.10],    # one      (x^5)
         [0.05, 0.80, 0.55]]    # step     (x^6)
    )

    # TODO 计算一个上下文向量 z
    x_2 = inputs[1]             # second input element              1×3
    d_in = inputs.shape[1]      # the input embedding size, dim=3     一般输入与输出维度相同,此处便于计算所以不同
    d_out = 2                   # the output embedding size, dim=2

    # 初始化权重
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 查询权重 均匀分布的随机张量,不会被优化器更新
    W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)    # 键权重
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)  # 值权重

    # 矩阵乘法
    query_2 = x_2 @ W_query     # (1×3) (3×2)  =  (1×2) torch.Size([2]) torch.Size([3]) torch.Size([3, 2])
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print(query_2)              # tensor([0.4306, 1.4551])   torch.Size([2])

    keys = inputs @ W_key       # torch.Size([6, 2])
    values = inputs @ W_value   # torch.Size([6, 2])

    # 计算注意力分数
    keys_2 = keys[1]            # 第2个输入词元的注意力分数
    attn_score_22 = query_2.dot(keys_2)
    print(attn_score_22)        # tensor(1.8524)    torch.Size([])

    attn_scores_2 = query_2 @ keys.T  # 查询词的所有注意分分数
    print(attn_scores_2)        # tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440]) torch.Size([6])

    # 将注意力分数转为注意力权重
    d_k = keys.shape[-1]        # 获取keys数组最后一个维度的大小 keys:torch.Size([6, 2]) d_k=2
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k ** 0.5, dim=-1)  # 注意缩放方式
    print(attn_weights_2)       # tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])
    print(keys.shape,keys,d_k)

    # 计算上下文向量
    context_vec_2 = attn_weights_2 @ values
    print(context_vec_2)        # tensor([0.3061, 0.8210]) torch.Size([2])

    # TODO 计算输入序列的所有上下文向量
    torch.manual_seed(123)      # 注意随机种子必须在此处,否则所有序列的上下文向量与单一词元的上下文向量对不上
    sa_v1 = SelfAttention_v1(d_in, d_out)
    output = sa_v1(inputs)
    print(output,output.shape)
    """
    tensor([[0.2996, 0.8053],
            [0.3061, 0.8210],
            [0.3058, 0.8203],
            [0.2948, 0.7939],
            [0.2927, 0.7891],
            [0.2990, 0.8040]], grad_fn=<MmBackward0>)  torch.Size([6, 2])
    """

    torch.manual_seed(123)      # 注意随机种子必须在此处,否则所有序列的上下文向量与单一词元的上下文向量对不上
    sa_v2 = SelfAttention_v2(d_in, d_out)
    output = sa_v2(inputs)
    print(output,output.shape)
    """
    tensor([[-0.5337, -0.1051],
            [-0.5323, -0.1080],
            [-0.5323, -0.1079],
            [-0.5297, -0.1076],
            [-0.5311, -0.1066],
            [-0.5299, -0.1081]], grad_fn=<MmBackward0>) torch.Size([6, 2])
    
    注意:
        两种模型初始化权重的方式不同,导致两者输出结果也不同    
            SelfAttention_v2使用nn.Linear 
            SelfAttention_v1使用torch.nn.Parameter(torch.rand(d_in, d_out))
    """

    # Q:将SelfAttention_v2的权重分配给SelfAttention_v1,查看二者输出相同
    sa_v1.W_query = torch.nn.Parameter(sa_v2.W_query.weight.T)
    sa_v1.W_key = torch.nn.Parameter(sa_v2.W_key.weight.T)
    sa_v1.W_value = torch.nn.Parameter(sa_v2.W_value.weight.T)
    output = sa_v1(inputs)
    print(output, output.shape)
    """
     tensor([[-0.5337, -0.1051],
            [-0.5323, -0.1080],
            [-0.5323, -0.1079],
            [-0.5297, -0.1076],
            [-0.5311, -0.1066],
            [-0.5299, -0.1081]], grad_fn=<MmBackward0>) torch.Size([6, 2])   
    """