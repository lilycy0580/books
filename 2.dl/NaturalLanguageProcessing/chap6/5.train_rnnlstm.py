import matplotlib.pyplot as plt
from chap6.data import ptb
from chap6.lstm.rnnlm import Rnnlm
from chap6.optimizer.optimizer import SGD
from chap6.train.rnnlm_trainer import RnnlmTrainer
from chap6.common.eval_perplexity import eval_perplexity

if __name__ == '__main__':
    # 设定超参数
    batch_size = 20
    wordvec_size = 100
    hidden_size = 100  # RNN的隐藏状态向量的元素个数
    time_size = 35  # RNN的展开大小
    lr = 20.0
    max_epoch = 4

    max_grad = 0.25

    # 读入训练数据
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    corpus_test, _, _ = ptb.load_data('test')
    vocab_size = len(word_to_id)
    xs = corpus[:-1]
    ts = corpus[1:]

    # 生成模型
    model = Rnnlm(vocab_size, wordvec_size, hidden_size)
    optimizer = SGD(lr)
    trainer = RnnlmTrainer(model, optimizer)

    # 应用梯度裁剪进行学习
    trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
    trainer.plot(ylim=(0, 500))
    plt.savefig('train_rnnlm.png')

    # 基于测试数据进行评价
    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test)
    print('test perplexity: ', ppl_test)            # test perplexity:  135.73865

    # 保存参数
    model.save_params()

    # | epoch 1 |  iter 1 / 1327 | time 1[s] | perplexity 10001.80
    # | epoch 4 |  iter 1321 / 1327 | time 424[s] | perplexity 109.78
    # 2017年的研究,PTB数据集上的困惑度已经降到60以下
