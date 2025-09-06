
import re

# 读取文本并分词
def process_text():
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    return preprocessed

# 构建简单的文本分词器
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i: s for s, i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]    # 遍历preprocessed中每个字符串元素s,并在self.str_to_int中找到对应值
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text) # 移除标点符号前面的多余空白字符
        return text

"""
    将词元转换为词元ID 将词元从字符串转为整数,以生成词元ID(token ID)
"""
if __name__ == '__main__':

    # 文本分词
    preprocessed = process_text()
    all_words = sorted(set(preprocessed))   # 获取词汇表
    print(all_words[100:110],len(all_words))
    """
    ['Thwing', 'Thwings', 'To', 'Usually', 'Venetian', 'Victor', 'Was', 'We', 'Well', 'What']   1130
    """

    # 创建词汇表
    vocab = {token: integer for integer, token in enumerate(all_words)}
    for i, item in enumerate(vocab.items()):
        print(item)
        if i >= 5:
            break
    """
    ('!', 0)
    ('"', 1)
    ("'", 2)
    ('(', 3)
    (')', 4)
    (',', 5)   
    """

    # 词汇表       将新文本转化为词元ID     encode
    # 逆向词汇表    将词元ID映射为文本词元   decode
    tokenizer = SimpleTokenizerV1(vocab)
    text = """ It's the last he painted, you know, Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    txt = tokenizer.decode(ids)
    text = tokenizer.decode(tokenizer.encode(text))
    print(ids)
    print(txt)
    print(text)
    """
    [56, 2, 850, 988, 602, 533, 746, 5, 1126, 596, 5, 67, 7, 38, 851, 1108, 754, 793, 7]
    It' s the last he painted, you know, Mrs. Gisburn said with pardonable pride.
    It' s the last he painted, you know, Mrs. Gisburn said with pardonable pride.
    """

    # 将分词器应用于训练集之外的新文本
    text = "Hello, do you like tea?"
    ids = tokenizer.encode(text)
    print(ids)
    """
    KeyError: 'Hello'       Hello一词并未在短篇小说The Verdict中出现
    """







