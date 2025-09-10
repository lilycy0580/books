
import torch as t
import torch.nn as nn
import matplotlib.pyplot as plt

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + t.tanh(t.sqrt(t.tensor(2.0 / t.pi)) *  (x + 0.044715 * t.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(                                    # 维度先升,激活后,再降,允许模型探索丰富的表示空间
            nn.Linear(cfg["embedding_dim"], 4 * cfg["embedding_dim"]),  # 768,768*4
            GELU(),
            nn.Linear(4 * cfg["embedding_dim"], cfg["embedding_dim"]))  # 768*4,768

    def forward(self, x):
        return self.layers(x)

"""
    实现具有GELU激活函数的前馈神经网络    
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # a:绘制relu与gelu的曲线图
    x = t.linspace(-3, 3, 100)      # 在[-3,3]之间创建100个样本点,绘制激活函数
    gelu, relu = GELU(), nn.ReLU()
    y_gelu, y_relu = gelu(x), relu(x)
    plt.figure(figsize=(8, 3))
    for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
        plt.subplot(1, 2, i)
        plt.plot(x, y)
        plt.title(f"{label} activation function")
        plt.xlabel("x")
        plt.ylabel(f"{label}(x)")
        plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("./gelu_relu.png")

    # b:使用GELU函数实现小型神经网络模块 FeedForward,该模块将在大语言模型的Transformer块使用
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # 词汇表大小     BPE分词器使用的由50257个单词组成的词汇表
        "context_length": 1024,     # 上下文长度     模型通过位置嵌入能够处理的最大输入词元数量
        "embedding_dim": 768,             # 嵌入维度       将每个词元转为768维的向量
        "n_heads": 12,              # 注意力头的数量  多头注意力机制中注意力头的数量
        "n_layers": 12,             # 层数          模型中的Transformer块数量
        "drop_rate": 0.1,           # dropout率     dropout机制的强制 0.1表示有10%的隐藏单元被随机丢弃,防止过拟合
        "qkv_bias": False           # 查询-键-值偏置  是否在多头注意力机制的线性层中添加一个偏置向量,用于查询,键和值的计算 刚开始禁用
    }
    print(GPT_CONFIG_124M["embedding_dim"])
    """
    768     每个词元的嵌入维度是768
    """

    ffn = FeedForward(GPT_CONFIG_124M)
    x = t.rand(2, 3, 768)
    out = ffn(x)
    print(out.shape)
    """
    torch.Size([2, 3, 768])         x:两个样本,每个样本有3个词元,每个词元是768维
    """