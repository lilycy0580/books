

import tiktoken     # 实现BPE算法
import importlib.metadata
"""
    BPE分词:
        1.BPE分词器
            BPE分词器可正确的编码与解码未知单词
            <|endoftext|> 词元被分配给一个较大的词元ID,50256 
                训练GPT-2,GPT-3,ChatGPT中使用的原始模型的BPE分词器的词汇量为50257
            
        2.BPE原理
            将不在预定义词汇表中的单词分解为更小的子词单元甚至单个字符,从而处理词汇表之外的单词
            若在分词过程中遇到不熟悉的单词,则可以将其表示为子词词元或字符序列,保证llm可以处理任何文本
"""
if __name__ == '__main__':

    version = importlib.metadata.version("tiktoken")
    print(f"tiktoken version: {version}")
    """
    tiktoken version: 0.7.0
    """

    # BPE分词
    text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.")
    tokenizer = tiktoken.get_encoding("gpt2")
    integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    strings = tokenizer.decode(integers)
    print(integers)
    print(strings)
    """
    [15496, 11, 466, 345, 588, 8887, 30, 220, 50256, 554, 262, 4252, 18250, 8812, 2114, 286, 617, 34680, 27271, 13]
    Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace.
    """





