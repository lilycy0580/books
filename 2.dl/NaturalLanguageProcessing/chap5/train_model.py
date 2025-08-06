# coding: utf-8

from chap5.gpu.np import *
from chap5.optimizer.optimizer import SGD
from chap5.data import ptb
from simple_rnnlm import SimpleRnnlm
from chap5.train.rnnlm_trainer import RnnlmTrainer

if __name__ == '__main__':
    # 设定超参数
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100  # RNN的隐藏状态向量的元素个数
    time_size = 5  # RNN的展开大小
    lr = 0.1
    max_epoch = 100

    # 读入训练数据
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000  # 缩小测试用的数据集
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)
    xs = corpus[:-1]  # 输入
    ts = corpus[1:]  # 输出（监督标签）

    # 生成模型
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    trainer.fit(xs, ts, max_epoch, batch_size, time_size)
    trainer.plot()
