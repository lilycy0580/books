# coding: utf-8

import matplotlib.pyplot as plt
from chap5.gpu.np import *
from chap5.optimizer.optimizer import SGD
from chap5.data import ptb
from simple_rnnlm import SimpleRnnlm

if __name__ == '__main__':
    # 设定超参数
    batch_size = 10
    wordvec_size = 100
    hidden_size = 100       # RNN的隐藏状态向量的元素个数
    time_size = 5           # Truncated BPTT的时间跨度大小
    lr = 0.1
    max_epoch = 100

    # 读入训练数据(缩小数据集)
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_size = 1000
    corpus = corpus[:corpus_size]
    vocab_size = int(max(corpus) + 1)

    xs = corpus[:-1]        # 输入
    ts = corpus[1:]         # 输出(监督标签)
    data_size = len(xs)
    print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

    # 学习用的参数
    max_iters = data_size // (batch_size * time_size)
    time_idx = 0
    total_loss = 0
    loss_count = 0
    ppl_list = []

    # 生成模型
    model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)

    # 计算读入mini-batch的各笔样本数据的开始位置
    jump = (corpus_size - 1) // batch_size
    offsets = [i * jump for i in range(batch_size)]

    for epoch in range(max_epoch):
        for iter in range(max_iters):
            # 获取mini-batch  数据需要按顺序输入,并且 mini-batch 的各批次要平移读入数据的开始位置
            batch_x = np.empty((batch_size, time_size), dtype='i')
            batch_t = np.empty((batch_size, time_size), dtype='i')
            for t in range(time_size):
                for i, offset in enumerate(offsets):
                    batch_x[i, t] = xs[(offset + time_idx) % data_size]
                    batch_t[i, t] = ts[(offset + time_idx) % data_size]
                time_idx += 1

            # 计算梯度,更新参数
            loss = model.forward(batch_x, batch_t)
            model.backward()
            optimizer.update(model.params, model.grads)
            total_loss += loss
            loss_count += 1

        # 各个epoch的困惑度评价  随着学习的进行,困惑度稳步下降
        ppl = np.exp(total_loss / loss_count)
        print('| epoch %d | perplexity %.2f'% (epoch + 1, ppl))
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

    # 绘制图形
    x = np.arange(len(ppl_list))
    plt.plot(np.asnumpy(x), np.asnumpy(ppl_list), label='train')
    plt.xlabel('epochs')
    plt.ylabel('perplexity')
    plt.show()
    plt.savefig('train_perplexity.png')

