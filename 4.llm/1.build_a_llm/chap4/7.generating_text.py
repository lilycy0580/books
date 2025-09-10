import tiktoken
import torch as t
import torch.nn as nn
from model import GPTModel,MultiHeadAttention, LayerNorm, FeedForward

def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    :param model:
    :param idx:             encoded_tensor  torch.Size([1, 4])
    :param max_new_tokens:  6                                       GPT模型生成序列的最大长度
    :param context_size:    GPT_CONFIG_124M["context_length"])      上下文最大长度
    """
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with t.no_grad():
            logits = model(idx_cond)                            # torch.Size([1, 4, 50257])
        logits = logits[:, -1, :]                               # torch.Size([1, 4, 50257])--->torch.Size([1, 50257])

        probability = t.softmax(logits, dim=-1)                 # torch.Size([1, 50257])    (batch, vocab_size)
        idx_next = t.argmax(probability, dim=-1, keepdim=True)  # torch.Size([1, 1])        (batch, 1)
        # idx_next = t.argmax(logits, dim=-1, keepdim=True)     贪心解码

        idx = t.cat((idx, idx_next), dim=1)             # torch.Size([1, 5])         (batch, n_tokens+1)
    return idx

# demo4.3
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            dim_in=cfg["embedding_dim"],
            dim_out=cfg["embedding_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate_attn"],          # NEW: dropout for multi-head attention 多头注意力
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["embedding_dim"])
        self.norm2 = LayerNorm(cfg["embedding_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate_shortcut"])  # 快捷连接 dropout

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x


class GPTModelDemo(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["embedding_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["embedding_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate_emb"])    # NEW: dropout for embedding layers 嵌入层

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["embedding_dim"])
        self.out_head = nn.Linear(cfg["embedding_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(t.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

"""
    生成文本
"""
if __name__ == '__main__':
    t.manual_seed(123)

    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # 词汇表大小     BPE分词器使用的由50257个单词组成的词汇表
        "context_length": 1024,     # 上下文长度     模型通过位置嵌入能够处理的最大输入词元数量
        "embedding_dim": 768,       # 嵌入维度       将每个词元转为768维的向量
        "n_heads": 12,              # 注意力头的数量  多头注意力机制中注意力头的数量
        "n_layers": 12,             # 层数          模型中的Transformer块数量
        "drop_rate": 0.1,           # dropout率     dropout机制的强制 0.1表示有10%的隐藏单元被随机丢弃,防止过拟合
        "qkv_bias": False           # 查询-键-值偏置  是否在多头注意力机制的线性层中添加一个偏置向量,用于查询,键和值的计算 刚开始禁用
    }

    start_context = "Hello, I am"

    tokenizer = tiktoken.get_encoding("gpt2")       # tiktoken分词器对批输入进行分词处理
    encoded = tokenizer.encode(start_context)
    encoded_tensor = t.tensor(encoded).unsqueeze(0)
    print("encoded:", encoded,"encoded_tensor.shape:", encoded_tensor.shape)
    """
    encoded: [15496, 11, 314, 716]
    encoded_tensor.shape: torch.Size([1, 4])
    """

    model = GPTModel(GPT_CONFIG_124M)
    model.eval()                                    # 禁用dropout等只在训练器件使用的随机组件
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=GPT_CONFIG_124M["context_length"])
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print("Output:", out)
    print("Output length:", len(out[0]))
    print(decoded_text)
    """
    Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267]])
    Output length: 10
    Hello, I am Featureiman Byeswickattribute argue 
    模型不能生成连贯文本的原因是没有对其进行训练,进实现GPT架构,并用初始随机权重初始化GPT模型
    """

    # demo4.3:为GPT模型不同dropout层指定不同的dropout值 嵌入层/快捷连接层/多头注意力模块
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 1024,
        "embedding_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate_emb": 0.1,       # NEW: dropout for embedding layers     嵌入层
        "drop_rate_attn": 0.1,      # NEW: dropout for multi-head attention 多头注意力模块
        "drop_rate_shortcut": 0.1,  # NEW: dropout for shortcut connections 快捷连接层
        "qkv_bias": False
    }
    model = GPTModelDemo(GPT_CONFIG_124M)
    print("Model:", model)
    """
    Model: GPTModelDemo(
      (tok_emb): Embedding(50257, 768)
      (pos_emb): Embedding(1024, 768)
      (drop_emb): Dropout(p=0.1, inplace=False)
      (trf_blocks): Sequential(
        (0): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (1): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (2): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (3): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (4): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (5): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (6): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (7): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (8): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (9): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (10): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
        (11): TransformerBlock(
          (att): MultiHeadAttention(
            (W_query): Linear(in_features=768, out_features=768, bias=False)
            (W_key): Linear(in_features=768, out_features=768, bias=False)
            (W_value): Linear(in_features=768, out_features=768, bias=False)
            (output_projection): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (ff): FeedForward(
            (layers): Sequential(
              (0): Linear(in_features=768, out_features=3072, bias=True)
              (1): GELU()
              (2): Linear(in_features=3072, out_features=768, bias=True)
            )
          )
          (norm1): LayerNorm()
          (norm2): LayerNorm()
          (drop_shortcut): Dropout(p=0.1, inplace=False)
        )
      )
      (final_norm): LayerNorm()
      (out_head): Linear(in_features=768, out_features=50257, bias=False)
    )    
    """