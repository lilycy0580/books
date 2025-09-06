
import tiktoken
import torch as t
from torch.utils.data import Dataset, DataLoader

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
    使用滑动窗口进行数据采样
        1.生成用于训练的输入——目标对,llm通过预测文本序列的下一个单词进行预训练
        
        2.数据加载器
            返回两个张量 一个为输入张量,一个为预测的目标张量  
        
"""
if __name__ == '__main__':
    # 使用滑动窗口的方法从训练集中提取输入——目标对
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    tokenizer = tiktoken.get_encoding("gpt2")
    encode_text = tokenizer.encode(raw_text)
    print(len(encode_text))
    """
    5145  使用BPE分词器后训练集中的词元总数
    """

    encode_sample = encode_text[50:]        # 从数据集中移除前50个

    context_size = 4                        # 上下文表示输入中包含4个词元
    x = encode_sample[:context_size]        # x 输入的词元
    y = encode_sample[1:context_size + 1]   # y 由x的每个输入词元右移一个位置所得的目标词元
    print(f"x: {x}")
    print(f"y:      {y}")
    """
    x: [290, 4920, 2241, 287]
    y:      [4920, 2241, 287, 257]
    """

    # 下一单词预测任务
    for i in range(1, context_size + 1):
        context = encode_sample[:i]
        desired = encode_sample[i]
        print(context, "---->", desired)
    """
    [290] ----> 4920
    [290, 4920] ----> 2241
    [290, 4920, 2241] ----> 287
    [290, 4920, 2241, 287] ----> 257        llm的输入 ----> llm预测的目标词元ID
    """

    for i in range(1, context_size + 1):
        context = encode_sample[:i]
        desired = encode_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    """
     and ---->  established
     and established ---->  himself
     and established himself ---->  in
     and established himself in ---->  a    llm训练的输入——目标对创建ok
    """

    # 使用batch_size=1的Dataloader对上下文长度为4的llm进行测试
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)    # 将dataloader转换为python迭代器,通过next()获取下一个条目
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)
    """
    [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])] 
    [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])] 
    注:
        [输入词元ID,目标词元ID] max_length=4(训练llm时,输入大小>=256)
        stride决定批次之间输入的位移量,模拟滑动窗口  
        从数据加载器中采样的批次=1,较小的批次大小会减少训练过程中的内存占用,但会导致模型在更新时产生更多的噪声 batch_size也是超参数
    """

    # batch_size>1使用数据加载器进行采样
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:", inputs)
    print("Targets:", targets)
    """
    Inputs: 
    tensor([[   40,   367,  2885,  1464],
            [ 1807,  3619,   402,   271],
            [10899,  2138,   257,  7026],
            [15632,   438,  2016,   257],
            [  922,  5891,  1576,   438],
            [  568,   340,   373,   645],
            [ 1049,  5975,   284,   502],
            [  284,  3285,   326,    11]])
    Targets: 
    tensor([[  367,  2885,  1464,  1807],
            [ 3619,   402,   271, 10899],
            [ 2138,   257,  7026, 15632],
            [  438,  2016,   257,   922],
            [ 5891,  1576,   438,   568],
            [  340,   373,   645,  1049],
            [ 5975,   284,   502,   284],
            [ 3285,   326,    11,   287]])
    注:
        stride=4 充分利用数据集(不会跳过任何一个单词),同时避免不同批次之间的数据重叠 过多重叠会增加模型的过拟合风险
    """











