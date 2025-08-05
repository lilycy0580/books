import collections
import numpy as np
from chap4.gpu.config import GPU


class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]  # 3

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)  # (3,2)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[
                    i]  # 1,3,0                   [0.18644244 0.         0.31355756 0.31355756 0.18644244]
                p[target_idx] = 0  # 正例的概率为0             [0.18644244 0.31355756 0.31355756 0.         0.18644244]
                p /= p.sum()  # 去掉正例后的所有负例的概率分布[0.         0.2781948  0.2781948  0.2781948  0.16541561]
                # 从vocab_size中按概率无放回的取sample_size个负例
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        else:
            # 在用GPU(cupy）计算时,优先速度   有时目标词存在于负例中
            negative_sample = np.random.choice(self.vocab_size,
                                               size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample





