
import torch as t
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 新添加的
        self.dropout = nn.Dropout(dropout)                                                              # dropout层
        self.register_buffer('mask', t.triu(t.ones(context_length, context_length), diagonal=1))  # 上三角掩码矩阵
        # register_buffer():当llm调使用此类时,缓冲区会与模型一起自动移动到适当的设备(cpu/gpu) 无需手动确保这些张量和模型参数在同一设备上

    def forward(self, x):
        batch_size, num_tokens, dim_in = x.shape    # batch_size,num_tokens,dim_in
                                                    # 批次大小,一个文本对应几个词元数,每个词元是几维的嵌入向量
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2) # 转置 (b,n,d)-->(b,d,n)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -t.inf)
        """
        因果掩码:
            masked_fill_()  
                将mask中为True的位置替换为指定的value
            
            self.mask.bool()[:num_tokens, :num_tokens]
                self.mask.bool()            预先注册好的因果掩码转化为布尔类型  上三角矩阵(对角线以上为1)
                [:num_tokens, :num_tokens]  只取前num_tokens行和num_tokens列
        """

        attn_weights = t.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vector = attn_weights @ values
        return context_vector

"""
    利用因果注意力隐藏未来词汇
        1.因果注意力的掩码实现
        2.利用dropout掩码额外的注意力权重   有效减少过拟合
        3.实现一个简化的因果注意力类
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # 3.实现一个简化的因果注意力类
    text = "Your journey starts with one step"
    inputs = t.tensor([[0.43, 0.15, 0.89],    # Your     (x^1)
                       [0.55, 0.87, 0.66],    # journey  (x^2)
                       [0.57, 0.85, 0.64],    # starts   (x^3)
                       [0.22, 0.58, 0.33],    # with     (x^4)
                       [0.77, 0.25, 0.10],    # one      (x^5)
                       [0.05, 0.80, 0.55]])   # step     (x^6)

    d_in = inputs.shape[1]  # 输入嵌入维度 d_in=3
    d_out = 2               # 输出嵌入维度 d_out=2

    batch = t.stack((inputs, inputs), dim=0)    # 复制文本模拟批量输入
    print(batch.shape)
    """
    torch.Size([2, 6, 3])   两个输入文本,每个文本有6个词元,每个词元是一个三维的嵌入向量
    """

    context_length = batch.shape[1]
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vectors = ca(batch)
    print(context_vectors,context_vectors.shape)
    """     
    tensor([[[-0.4519,  0.2216],
             [-0.5874,  0.0058],
             [-0.6300, -0.0632],
             [-0.5675, -0.0843],
             [-0.5526, -0.0981],
             [-0.5299, -0.1081]],
    
            [[-0.4519,  0.2216],
             [-0.5874,  0.0058],
             [-0.6300, -0.0632],
             [-0.5675, -0.0843],
             [-0.5526, -0.0981],
             [-0.5299, -0.1081]]], grad_fn=<UnsafeViewBackward0>)       torch.Size([2, 6, 2])
    """
