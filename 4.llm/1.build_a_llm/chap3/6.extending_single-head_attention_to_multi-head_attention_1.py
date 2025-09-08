
import torch as t
import torch.nn as nn


class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 新添加的
        self.dropout = nn.Dropout(dropout)  # dropout层
        self.register_buffer('mask', t.triu(t.ones(context_length, context_length), diagonal=1))  # 上三角掩码矩阵
        # register_buffer():当llm调使用此类时,缓冲区会与模型一起自动移动到适当的设备(cpu/gpu) 无需手动确保这些张量和模型参数在同一设备上

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape  # batch_size,num_tokens,dim_in
        # 批次大小,一个文本对应几个词元数,每个词元是几维的嵌入向量
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)  # 转置 (b,n,d)-->(b,d,n)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -t.inf)
        """
        因果掩码:
            masked_fill_()  
                将mask中为True的位置替换为指定的value

            self.mask.bool()[:num_tokens, :num_tokens]
                self.mask.bool()            预先注册好的因果掩码转化为布尔类型  上三角矩阵(对角线以上为1)
                [:num_tokens, :num_tokens]  只取前num_tokens行和num_tokens列
        """

        attn_weights = t.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values
        return context_vector

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])
        """
        ModuleList(
          (0-1): 2 x CausalAttention(
            (W_query): Linear(in_features=3, out_features=2, bias=False)
            (W_key): Linear(in_features=3, out_features=2, bias=False)
            (W_value): Linear(in_features=3, out_features=2, bias=False)
            (dropout): Dropout(p=0.0, inplace=False)
          )
        )
        """

    def forward(self, x):
        return t.cat([head(x) for head in self.heads], dim=-1)
    """
        输入x ——> head1 ——> 输出1
                                ——> 拼接 ——> 最终输出 
        输入x ——> head2 ——> 输出2   
        
        outputs = []
        for head in self.heads:
            output = head(x)            torch.Size([2, 6, 2])
            outputs.append(output)  
        res = t.cat(outputs, dim=-1)    torch.Size([2, 6, 4])
    """

"""
    将单头注意力扩展到多头注意力
        1.叠加多个单头注意力层
        2.通过权重划分实现多头注意力
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # 1.叠加多个单头注意力层
    text = "Your journey starts with one step"
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3
    d_out = 2               # 输出嵌入维度 d_out=2

    batch = t.stack((inputs, inputs), dim=0)    # 按batch输入
    context_length = batch.shape[1]                     # 一个输入元素对应的词元数量
    d_in, d_out = 3, 2
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vectors = mha(batch)
    print(context_vectors, context_vectors.shape)
    """
    tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
             [-0.5874,  0.0058,  0.5891,  0.3257],
             [-0.6300, -0.0632,  0.6202,  0.3860],
             [-0.5675, -0.0843,  0.5478,  0.3589],
             [-0.5526, -0.0981,  0.5321,  0.3428],
             [-0.5299, -0.1081,  0.5077,  0.3493]],
    
            [[-0.4519,  0.2216,  0.4772,  0.1063],
             [-0.5874,  0.0058,  0.5891,  0.3257],
             [-0.6300, -0.0632,  0.6202,  0.3860],
             [-0.5675, -0.0843,  0.5478,  0.3589],
             [-0.5526, -0.0981,  0.5321,  0.3428],
             [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>) 
    torch.Size([2, 6, 4])   
        2个输入文本 每个输入中有6个词元 每个词元的4维嵌入
    """

    # demo:保证输出的上下文向量是二维而不是四维 同时保持设置num_heads=2
    d_out = 1
    mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
    context_vectors = mha(batch)
    print(context_vectors.shape)
    """
    torch.Size([2, 6, 2])
    """