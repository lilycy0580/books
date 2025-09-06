
import re

# 文本分词
def process_text():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed

# 处理未知单词的文本分词器
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text

"""
    引入特殊上下文词元   
        处理未知的单词,引入特殊上下文词元,增加llm对上下文和其他相关信息的理解 特殊词元可能包括用于标识未知词汇和文档边界的词元
        
        修改词汇表和分词器,支持新词元<unk><endoftext>
            遇到词汇表中不存在的单词时,使用特殊词元<unk>代替
            不相关的文本之间插入特殊词元<endoftext> eg:训练类GPT的模型时,在每个文档或图书的开头添加<endoftext>区分前一个文本源
            
        不同的llm引入不同的特殊词元:
            [BOS] 标记文本的起点
            [EOS] 位于文本的末尾,连接多个不相关的文本
            [PAD] 当使用batch>1的批次数据训练llm时,数据中的文本长度不同,添加[PAD]进行扩展或填充
        
        GPT模型仅使用<|endoftext|>简化处理流程,也用于文本的填充
        GPT模型使用BPE分词器将单词拆分为子词单元  
"""
if __name__ == '__main__':
    preprocessed = process_text()
    all_tokens = sorted(set(preprocessed))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token: integer for integer, token in enumerate(all_tokens)}
    print(len(vocab.items()))
    """
    1132    更新后的词汇表为1132 之前是1130 
    """

    for i, item in enumerate(list(vocab.items())[-5:]):
        print(item)
    """
    ('younger', 1127)
    ('your', 1128)
    ('yourself', 1129)
    ('<|endoftext|>', 1130)         输出最后5个条目,发现新增2个词元添加到词汇表中
    ('<|unk|>', 1131)
    """

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."
    text = " <|endoftext|> ".join((text1, text2))
    print(text)
    """
    Hello, do you like tea? <|endoftext|> In the sunlit terraces of the palace.
    """

    tokenizer = SimpleTokenizerV2(vocab)
    ids = tokenizer.encode(text)
    print(ids)
    """
    [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]
    """

    text = tokenizer.decode(ids)
    print(text)
    """
    <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.
    确定训练数据集缺少Hello与palace两个单词
    """
























