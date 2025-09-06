
import tiktoken
import torch as t
from torch.utils.data import DataLoader, Dataset

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # 对全部文本进行分词
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # 使用滑动窗口将文本划分为长度为max_length的重叠序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i+max_length]
            target_chunk = token_ids[i+1 : i+max_length+1]
            self.input_ids.append(t.tensor(input_chunk))
            self.target_ids.append(t.tensor(target_chunk))

    # 返回数据集总行数
    def __len__(self):
        return len(self.input_ids)

    # 返回数据集的指定行
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

# 用于批量生成输入——目标对的数据加载器 上下文长度为4
def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")                   # 初始化分词器
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)  # 创建数据集
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,drop_last=drop_last,num_workers=num_workers)
    return dataloader
    # drop_last=True且批次大小小于指定的batch_size,则会删除最后一批,防止在训练中出现损失剧增
    # num_workers预处理CPU的进程数

"""
    编码单词位置信息
        1.词元嵌入适合作为llm的输入,但是llm的自注意力机制无法感知词元在序列中的位置或顺序
            嵌入层的工作原理是无论词元ID在输入序列的位置如何,相同的词元ID被映射到相同的向量表示
            llm的自注意力机制本质上与位置无关,故向模型中注入额外的位置信息
            
        2.位置信息嵌入
            绝对位置嵌入:
                直接与序列中的特定位置相关联 对于输入序列中的每个位置,均会在对应词元的嵌入向量中添加一个位置嵌入,指明在序列中的位置
                (OpenAI的GPT模型使用绝对位置嵌入,这些嵌入会在训练过程中优化,有别于原始Transformer中的固定或预定义位置编码)
                
            相对位置嵌入:
                关注词元之间的相对位置或距离,而非绝对位置 模型学习的是词元之间的"距离"关系,而非在序列中的"具体位置"
                (方便模型更好适用不同长度的序列)
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.获取词元嵌入向量
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    # 从数据加载器中采样数据时,每批次中每个词元被嵌入一个256维的向量
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = t.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:", inputs)
    print("Inputs shape:", inputs.shape)
    """
    Token IDs: 
    tensor([[   40,   367,  2885,  1464],
            [ 1807,  3619,   402,   271],
            [10899,  2138,   257,  7026],
            [15632,   438,  2016,   257],
            [  922,  5891,  1576,   438],
            [  568,   340,   373,   645],
            [ 1049,  5975,   284,   502],
            [  284,  3285,   326,    11]])
    Inputs shape: torch.Size([8, 4])                词元ID张量为8*4,8个样本,每个样本由4个词元组成
    """

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)
    """
    torch.Size([8, 4, 256])
    注:
        使用嵌入层将词元ID嵌入256维的向量中 8*4*256 表示每个词元ID被嵌入一个256维的向量中
    """

    # 2.获取绝对位置嵌入  GPT模型采用绝对位置嵌入
    context_length = max_length
    pos_embedding_layer = t.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(t.arange(max_length))
    print(t.arange(max_length))
    print(pos_embeddings.shape)
    """
    torch.Size([4, 256])    位置嵌入由4个256维的向量组成,可将这些向量直接添加到词元嵌入中
    注:
        pos_embedding_layer
            输入是个占位符向量,tensor([0, 1, 2, 3]) 从0开始递增,直到最大输入长度减1的数值序列
            
        context_length 表示模型支持的输入块的最大长度 本文设置与输入文本的最大长度一致
            实际使用中,输入文本的长度可能会超出模型支持的块大小,此时需截断文本
    """

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    """
    torch.Size([8, 4, 256]) 词元嵌入向量+绝对位置嵌入 可直接作为llm核心模块处理的嵌入输入
    """

