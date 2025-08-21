import re
import urllib
from importlib.metadata import version
import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int
            else "<|unk|>" for item in preprocessed
        ]

        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers)
    return dataloader


if __name__ == '__main__':
    # TODO 2.2 文本分词
    # 从github上下载文本 the-verdict.txt  20479个字符
    url = ("https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

    # python标准文件读取工具加载 the-verdict.txt
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))                              # 20479
    print(raw_text[:99])

    # 输入文本————>词元化文本
    # 将the-verdict.txt分割为独立的单词和特殊符号,方便后续转为嵌入向量,用于llm训练
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(len(preprocessed))                                                        # 4690
    print(preprocessed[:30])

    # TODO 2.3 将词元转换为词元ID
    all_words = sorted(set(preprocessed))
    vocab_size = len(all_words)
    print(vocab_size)                                                               # 1130 词元去重后总个数

    # 词汇表
    vocab = {token:integer for integer, token in enumerate(all_words)}              # 创建字典dict
    for i, item in enumerate(vocab.items()):                                        # {"hello":0, "world":1, "python":2}
        print(item)
        if i >= 50:
            break

    tokenizer = SimpleTokenizerV1(vocab)
    text = """"It's the last he painted, you know,"
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)                                                    # [1, 56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 1, 67, 7, 38, 851, 1108, 754, 793, 7]
    txt = tokenizer.decode(ids)
    text = tokenizer.decode(tokenizer.encode(text))
    print(ids,"\n",txt,"\n",text)

    # text = "Hello, do you like tea?"                                                # 报错,因为hello没有在词汇表中
    # print(tokenizer.encode(text))                                                   # 出现未知单词,需对分词器进行修改

    # TODO 2.4 引入特殊上下文词元
    # 将未知词元添加到词汇表中
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab.items()))

    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)

    # 修改后的分词器 添加 <|endoftext|> 与 <|unk|>
    tokenizer = SimpleTokenizerV2(vocab)
    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    ids = tokenizer.encode(text)                                                    #  The Verdict,包含“Hello”和“palace”
    text = tokenizer.decode(tokenizer.encode(text))
    print(ids,"\n",text)

    # TODO 2.5 BPE
    from importlib.metadata import version
    import tiktoken
    try:
        pkg_version = version("tiktoken")
        print(f"tiktoken 版本: {pkg_version}")                                        # tiktoken版本: 0.7.0
    except Exception as e:
        print(f"错误: {e}")

    tokenizer = tiktoken.get_encoding("gpt2")
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    strings = tokenizer.decode(integers)
    print(integers)
    print(strings)
    # [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
    # Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
    """
        1.<|endoftext|>词元被分配了一个较大的词元ID,50256.
            BPE分词器的词汇总量为50257,<|endoftext|>被分配了最大的词元ID
        2.BPE分词器可以正确地编码和解码未知单词,比如"someunknownPlace"
    """

    # TODO 2.6 使用滑动窗口进行数据采样
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))                                                            # BPE分词后的词元个数 5145

    enc_sample = enc_text[50:]                                                      # 去除前50个词元
    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size + 1]
    print(f"x: {x}")
    print(f"y:      {y}")
    """
    x: [290, 4920, 2241, 287]
    y:      [4920, 2241, 287, 257]
    """

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)
    """
    [290] ----> 4920
    [290, 4920] ----> 2241
    [290, 4920, 2241] ----> 287
    [290, 4920, 2241, 287] ----> 257
    """

    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    """
     and ---->  established
     and established ---->  himself
     and established himself ---->  in
     and established himself in ---->  a
    """

    # 本节使用Pytorch的Dataset与DataLoader
    print("PyTorch version:", torch.__version__)                                        # 2.5.0
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(raw_text, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)
    second_batch = next(data_iter)
    print(second_batch)
    """
    [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
    [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]
    """

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Inputs:\n", inputs)
    print("\nTargets:\n", targets)
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
    """


    # TODO 2.7 创建词元嵌入
    # 词元ID转为嵌入向量
    torch.manual_seed(123)
    vocab_size = 6                                                          # 词汇表:6个单词  嵌入维度:3
    output_dim = 3
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)            # 嵌入层
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))
    """
    tensor([[ 0.3374, -0.1778, -0.1690],
            [ 0.9178,  1.5810,  1.3010],
            [ 1.2753, -0.2010, -0.1606],
            [-0.4015,  0.9666, -1.1481],
            [-1.1589,  0.3255, -0.6315],
            [-2.8400, -0.7849, -1.4096]], requires_grad=True)
    tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)
    """

    # TODO 2.8 编码单词位置信息
    max_length = 4
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)      # 词元的嵌入向量
    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)                                           # torch.Size([8, 4, 256])

    # context_length:表示模型支持的输入块的最大长度,设为与输入文本的最大长度一致
    #                若输入文本的长度可能会超出模型支持的块大小,这时需要截断文本
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)                                             # torch.Size([4, 256])

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)                                           # torch.Size([8, 4, 256])
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
        
        Inputs shape:
         torch.Size([8, 4])    
    """
