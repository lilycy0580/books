import torch as t
import torch.nn as nn


if __name__ == '__main__':
    t.manual_seed(1000)

    # 随机输入张量 size=(2,3,4)
    # LSTM的输入 (sequence_length, batch_size, input_size)
    # 序列有2个时间步,批次中有3个样本,每个时间步的输入特征维度是4
    input = t.randn(2,3,4).float()

    # 构建一个lstm
    # input_size = 4
    # hidden_size = 3 即输出维度
    # num_layers = 1  只有1层lstm
    lstm = nn.LSTM(4,3,1)

    # 初始化隐藏状态 (num_layers, batch_size, hidden_size)
    # num_layers = 1   1层LSTM
    # batch_size = 3   3个样本
    # hidden_size = 3  隐藏状态维度为3
    h0 = t.randn(1,3,3).float()

    # 初始化细胞状态 (num_layers, batch_size, hidden_size) 同h0
    c0 = t.randn(1,3,3).float()

    # 将输入和初始状态传入LSTM层
    # out1 所有时间步的输出
    # hn   最后一个时间步的隐藏状态和细胞状态的元组
    out1,hn = lstm(input,(h0,c0))
    print(out1.size())
    """
    torch.Size([2, 3, 3])
    """

    input = t.randn(2,3,4).float()
    lstm = nn.LSTMCell(4,3)
    hx = t.randn(3,3).float()
    cx = t.randn(3,3).float()
    out = []
    for i in input:
        hx, cx = lstm(i, (hx, cx))
        out.append(hx)
    out2 = t.stack(out)
    print(out2.size())
    """
    torch.Size([2, 3, 3])
    """

    bool = out1.allclose(out2)
    print(bool)
    """
    False
    """

    # 词向量 生成词向量的Embedding层
    embedding = nn.Embedding(4,5)
    # 预训练好的词向量初始化embedding
    weight = t.arange(0,20).view(4,5).float()
    nn.Embedding.from_pretrained(weight)
    print(embedding)
    """
    Embedding(4, 5)
    """

    input = t.arange(3,0,-1).long()
    output = embedding(input)
    print(output)
    """
    tensor([[ 0.1533,  1.5595, -1.1288, -0.7424,  0.3131],
            [ 0.0521,  1.2395,  0.1319, -1.5994, -0.2877],
            [ 1.0873,  0.1332,  0.6201, -0.8361, -1.3498]],grad_fn=<EmbeddingBackward0>)
    """
