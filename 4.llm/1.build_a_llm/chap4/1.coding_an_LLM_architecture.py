
import torch as t
import torch.nn as nn
import tiktoken

class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

# 一个包含占位符的GPT模型架构类
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_embedding = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_embedding = nn.Dropout(cfg["drop_rate"])

        # 使用占位符替换TransformerBlock Transformer块
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        # 使用占位符替换LayerNorm 层归一化
        self.final_norm = DummyLayerNorm(cfg["embedding_dim"])

        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)                                 # 词元嵌入
        pos_embeddings = self.pos_embedding(t.arange(seq_len, device=in_idx.device))    # 位置嵌入
        x = token_embeddings + pos_embeddings

        x = self.drop_embedding(x)                                                      # dropout嵌入
        x = self.trf_blocks(x)                                                          # Transformer块
        x = self.final_norm(x)                                                          # 层归一化
        logits = self.out_head(x)                                                       # 线性输出层
        return logits


"""
    构建一个大预言模型架构
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # 1.构建一个大预言模型架构
    """
    只有解码器的Transformer语言模型:
        如何将输入的token索引转换为预测下一个token的概率分布
        输入索引-->词嵌入+位置嵌入-->Dropout-->Transformer块-->层归一化-->输出层-->预测logits 
    """

    # a:设置配置信息 dict类型
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # 词汇表大小     BPE分词器使用的由50257个单词组成的词汇表
        "context_length": 1024,     # 上下文长度     模型通过位置嵌入能够处理的最大输入词元数量
        "embedding_dim": 768,             # 嵌入维度       将每个词元转为768维的向量
        "n_heads": 12,              # 注意力头的数量  多头注意力机制中注意力头的数量
        "n_layers": 12,             # 层数          模型中的Transformer块数量
        "drop_rate": 0.1,           # dropout率     dropout机制的强制 0.1表示有10%的隐藏单元被随机丢弃,防止过拟合
        "qkv_bias": False           # 查询-键-值偏置  是否在多头注意力机制的线性层中添加一个偏置向量,用于查询,键和值的计算 刚开始禁用
    }

    tokenizer = tiktoken.get_encoding("gpt2")       # tiktoken分词器对批输入进行分词处理
    batch = []
    text1 = "Every effort moves you"
    text2 = "Every day holds a"
    batch.append(t.tensor(tokenizer.encode(text1)))
    batch.append(t.tensor(tokenizer.encode(text2)))
    batch = t.stack(batch, dim=0)
    print(batch)
    """
    tensor([[6109, 3626, 6100,  345],
            [6109, 1110, 6622,  257]])     获取词元ID
    """

    model = DummyGPTModel(GPT_CONFIG_124M)
    logits = model(batch)
    print(logits, logits.shape)
    """
    tensor([[[-0.9289,  0.2748, -0.7557,  ..., -1.6070,  0.2702, -0.5888],
             [-0.4476,  0.1726,  0.5354,  ..., -0.3932,  1.5285,  0.8557],
             [ 0.5680,  1.6053, -0.2155,  ...,  1.1624,  0.1380,  0.7425],
             [ 0.0447,  2.4787, -0.8843,  ...,  1.3219, -0.0864, -0.5856]],
    
            [[-1.5474, -0.0542, -1.0571,  ..., -1.8061, -0.4494, -0.6747],
             [-0.8422,  0.8243, -0.1098,  ..., -0.1434,  0.2079,  1.2046],
             [ 0.1355,  1.1858, -0.1453,  ...,  0.0869, -0.1590,  0.1552],
             [ 0.1666, -0.8138,  0.2307,  ...,  2.5035, -0.3055, -0.3083]]], grad_fn=<UnsafeViewBackward0>)
    
    torch.Size([2, 4, 50257])   
        两个文本,每个文本有4个词元,每个词元是一个50257维的向量(与分词表大小一致) 
        将50257维的向量转换回词元ID后,即可解码为单词
    """












