
from importlib.metadata import version
import os
import urllib.request
import re
"""
    文本分词 将the-verdict.txt 20479个字符分割为独立的单词和特殊字符,方便后续转为嵌入向量
"""
if __name__ == '__main__':
    print("torch version:", version("torch"))
    print("tiktoken version:", version("tiktoken"))
    """
    torch version: 2.5.0
    tiktoken version: 0.7.0
    """

    # 从git上下载并读取the-verdict.txt
    if not os.path.exists("the-verdict.txt"):
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        file_path = "./the-verdict.txt"
        urllib.request.urlretrieve(url, file_path)

    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))
    print(raw_text[:99])
    """
    Total number of character: 20479
    I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no
    """

    # 文本分割
    text = "Hello, world. This, is a test."
    result = re.split(r'(\s)', text)    # 按空白字符分割字符串,但会保留分隔符
    print(result)
    """
    ['Hello,', ' ', 'world.', ' ', 'This,', ' ', 'is', ' ', 'a', ' ', 'test.']   标点符合和单词在一起
    """

    result = re.split(r'([,.]|\s)', text)   # 按逗号,句点或任何空白字符来分割字符串,并且会保留这些分隔符
    print(result)
    """
    ['Hello', ',', '', ' ', 'world', '.', '', ' ', 'This', ',', '', ' ', 'is', ' ', 'a', ' ', 'test', '.', '']  
    标点符号和单词独立分割,但是有重复
    """

    result = [item for item in result if item.strip()]  # 过滤掉列表中的所有空字符串和只包含空白字符的字符串
    print(result)
    """
    ['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']   标点符号和单词独立分割,去除冗余字符与空白字符
    开发简易分词器时,空白字符是单独编码还是直接移除,取决于具体的场景和需求
    """

    text = "Hello, world. Is this-- a test?"
    result = re.split(r'([,.:;?_!"()\']|--|\s)', text)  # 按照多种标点符号,双破折号和空白字符来分割文本,并且会保留这些分隔符
    result = [item.strip() for item in result if item.strip()]
    print(result)
    """
    ['Hello', ',', 'world', '.', 'Is', 'this', '--', 'a', 'test', '?']
    """

    # 构建简易分词器,对the-verdict.txt进行分词
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print(preprocessed[:30])
    print(len(preprocessed))
    """
    ['I', 'HAD', 'always', 'thought', 'Jack', 'Gisburn', 'rather', 'a', 'cheap', 'genius', '--', 'though', 'a', 'good', 'fellow', 'enough', '--', 'so', 'it', 'was', 'no', 'great', 'surprise', 'to', 'me', 'to', 'hear', 'that', ',', 'in']
    4690
    """

