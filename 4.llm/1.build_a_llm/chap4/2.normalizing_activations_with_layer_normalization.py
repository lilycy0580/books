
import torch as t
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, embedding_dim):                      # embedding_dim:词嵌入的维度 对应输入张量x的最后一个维度
        super().__init__()
        self.eps = 1e-5                                     # 防止除以零的数值不稳定情况
        self.scale = nn.Parameter(t.ones(embedding_dim))    # 可学习的缩放参数 γ
        self.shift = nn.Parameter(t.zeros(embedding_dim))   # 可学习的平移参数 β

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)   # unbiased=False 嵌入维度n比较大的大语言模型,使用n和n-1的差异可以忽略
        norm_x = (x - mean) / t.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    """
    unbiased=True
        贝塞尔修正:样本方差的估计中使用n-1作为分母,调整偏差
    """

"""
    使用层归一化进行归一化激活
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # a:使用LayerNorm实现层归一化
    batch_example = t.randn(2, 5)                # 2个训练样本,每个样本5个维度
    layer = nn.Sequential(
        nn.Linear(5, 6),    # 1个线性层
        nn.ReLU())                               # 1个非线性激活函数 确保输入都是非负数
    out = layer(batch_example)
    print(out)
    """
    tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
            [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]], grad_fn=<ReluBackward0>)
    """

    mean = out.mean(dim=-1, keepdim=True)        # 归一化前的均值和方差  keepdim=True保证输出张量于输入张量具有相同的维度
    var = out.var(dim=-1, keepdim=True)
    print("Mean:", mean)
    print("Variance:", var)
    """
    Mean: tensor([[0.1324],[0.2170]], grad_fn=<MeanBackward1>)
    Variance: tensor([[0.0231],[0.0398]], grad_fn=<VarBackward0>)
    """

    out_norm = (out - mean) / t.sqrt(var)       # 层归一化操作 (输出值-均值)/标准差   标准差=方差 ** 0.5
    print("Normalized layer outputs:", out_norm)
    """
    Normalized layer outputs: 
    tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
            [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]], grad_fn=<DivBackward0>)
    """

    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Mean:", mean)
    print("Variance:", var)
    """
    Mean: tensor([[9.9341e-09],[0.0000e+00]], grad_fn=<MeanBackward1>)  因为计算机表示数值的有限精度存在数值误差,接近0,但非0
    Variance: tensor([[1.0000],[1.0000]], grad_fn=<VarBackward0>)       
    """

    t.set_printoptions(sci_mode=False)  # 关闭科学计数法
    print("Mean:", mean)
    print("Variance:", var)
    """
    Mean: tensor([[    0.0000],[    0.0000]], grad_fn=<MeanBackward1>)
    Variance: tensor([[1.0000],[1.0000]], grad_fn=<VarBackward0>)
    """

    # b:将层归一化封装成一个Pytorch模块
    laynorm = LayerNorm(embedding_dim=5)
    out_laynorm = laynorm(batch_example)
    mean = out_laynorm.mean(dim=-1, keepdim=True)
    var = out_laynorm.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:", mean)
    print("Variance:", var)
    """
    Mean: tensor([[    -0.0000],[     0.0000]], grad_fn=<MeanBackward1>)
    Variance: tensor([[1.0000],[1.0000]], grad_fn=<VarBackward0>)
    """