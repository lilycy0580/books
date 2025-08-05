import collections
import numpy as np
from chap4.cpu.config import GPU


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
        batch_size = target.shape[0]        # 3

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)  # (3,2)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]  # 1,3,0                   [0.18644244 0.         0.31355756 0.31355756 0.18644244]
                p[target_idx] = 0       # 正例的概率为0             [0.18644244 0.31355756 0.31355756 0.         0.18644244]
                p /= p.sum()            # 去掉正例后的所有负例的概率分布[0.         0.2781948  0.2781948  0.2781948  0.16541561]
                # 从vocab_size中按概率无放回的取sample_size个负例
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)

        else:
            # 在用GPU(cupy）计算时,优先速度
            # 有时目标词存在于负例中
            negative_sample = np.random.choice(self.vocab_size,
                                               size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


if __name__ == '__main__':
    np.random.seed(43)

    """
    choice:
        size        采样次数
        replace     是否放回
        p           基于概率分布采样
    """
    # 负采样的采样方法
    a = np.random.choice(10)
    b = np.random.choice(10)
    print(a,b)                                          # 6 3

    words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
    c = np.random.choice(words)
    print(c)                                            # hello

    d = np.random.choice(words, size=5)                 # 有放回
    e = np.random.choice(words, size=5, replace=False)  # 无放回
    p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
    f = np.random.choice(words, p=p)                    # 基于概率

    # 负采样 基于概率的0.75次方
    p = [0.7, 0.29, 0.01]
    new_p = np.power(p, 0.75)
    new_p /= np.sum(new_p)                              # [ 0.64196878 0.33150408 0.02652714]
    print(new_p)

    corpus = np.array([0, 1, 2, 3, 4, 1, 2, 3])
    power = 0.75
    sample_size = 2
    sampler = UnigramSampler(corpus, power, sample_size)
    print(sampler.sample_size,sampler.vocab_size,sampler.word_p)
    #       2                       5           [0.14193702 0.23870866 0.23870866 0.23870866 0.14193702]
    target = np.array([1, 3, 0])
    negative_sample = sampler.get_negative_sample(target)
    print(negative_sample)
    """
    正例:
        1,3,0
    负例:
        [[0 3]          正例1对应的2个负例样本
         [1 2]          正例3对应的2个负例样本
         [1 4]]         正例0对应的2个负例样本
    """

