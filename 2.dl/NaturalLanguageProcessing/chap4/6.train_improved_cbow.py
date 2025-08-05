import os

import matplotlib.pyplot as plt
import numpy as np

# 对数据的处理
from chap4.data import ptb
from chap4.gpu import config
from chap4.gpu.np import np
from chap4.gpu.gpu_data import create_contexts_target,to_gpu,to_cpu

# 加载模型及优化器
from chap4.model.cbow import CBOW
from chap4.optimizer.optimizer import Adam
from chap4.train.trainer import Trainer

# from chap4.common.unigram_sampler import UnigramSampler                 # 负采样
# from chap4.common.sigmoid_with_loss import SigmoidWithLoss              # 二分类 sigmoid with loss
# from chap4.common.embedding_dot import EmbeddingDot,Embedding           # Embedding+Dot
# from chap4.common.negative_sampling_loss import NegativeSamplingLoss    # 负采样loss

import pickle

# from skip_gram import SkipGram

if __name__ == '__main__':
    # 设定超参数
    window_size = 5
    hidden_size = 100
    batch_size = 100
    max_epoch = 10

    # 读入数据
    corpus, word_to_id, id_to_word = ptb.load_data('train')
    vocab_size = len(word_to_id)

    contexts, target = create_contexts_target(corpus, window_size)
    if config.GPU:
        contexts, target = to_gpu(contexts), to_gpu(target)

    # 生成模型等
    model = CBOW(vocab_size, hidden_size, window_size, corpus)
    # model = SkipGram(vocab_size, hidden_size, window_size, corpus)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)

    # 开始学习
    trainer.fit(contexts, target, max_epoch, batch_size)
    trainer.plot()
    plt.savefig("./train_improved_cbow_loss.jpg")

    # 保存必要数据,以便后续使用
    word_vecs = model.word_vecs         # 输入侧的权重
    if config.GPU:
        word_vecs = to_cpu(word_vecs)
    params = {}
    params['word_vecs'] = word_vecs.astype(np.float16)
    params['word_to_id'] = word_to_id
    params['id_to_word'] = id_to_word
    pkl_file = 'cbow_params.pkl'
    # pkl_file = 'skipgram_params.pkl'
    with open(pkl_file, 'wb') as f:     # pickle 文件保存
        pickle.dump(params, f, -1)

    """
    | epoch 1 |  iter 1 / 9295 | time 2[s] | loss 4.16
    | epoch 10 |  iter 9281 / 9295 | time 5041[s] | loss 1.47
    """

