import matplotlib.pyplot as plt
from chap3.config.np import *

from chap3.train.trainer import Trainer
from chap3.optimizer.optimizer import Adam
from chap3.net.simple_cbow import SimpleCBOW
from chap3.common.util import preprocess, create_contexts_target,convert_one_hot

if __name__ == '__main__':
    # CBOW模型的学习:准备数据,前向传播求梯度,更新权重参数
    # 超参数
    window_size = 1
    hidden_size = 5
    batch_size = 3
    max_epoch = 1000

    # 预处理
    text = 'You say goodbye and I say hello.'
    corpus, word_to_id, id_to_word = preprocess(text)

    # 词汇个数
    vocab_size = len(word_to_id)
    contexts, target = create_contexts_target(corpus, window_size)
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)

    # 构建cbow模型
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()
    plt.savefig("./loss.png")

    # 查看权重  输入侧的MatMul 层的权重
    word_vecs = model.word_vecs
    for word_id, word in id_to_word.items():
        print(word, word_vecs[word_id])
    """
    即单词的分布式表示:
        you     [ 1.2201523  1.297968  -0.9853682 -1.1124161 -1.0952643]
        say     [-1.201793    0.07621195  1.1814178   1.1800913   1.2311002 ]
        goodbye [ 0.57226086  0.694084   -0.95429236 -0.80009735 -0.73565966]
        and     [-1.0816752  -1.8935663   0.97765976  1.0314205   1.1401229 ]
        i       [ 0.5923486   0.70584047 -0.97637254 -0.786738   -0.7317797 ]
        hello   [ 1.2371646  1.2977684 -0.9931553 -1.103898  -1.0986371]
        .       [-0.9358754   1.662383    1.0440832   0.9643201   0.82404286]
    """