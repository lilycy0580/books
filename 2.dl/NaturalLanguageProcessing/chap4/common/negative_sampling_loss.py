from chap4.gpu.np import np
from chap4.common.unigram_sampler import UnigramSampler
from chap4.common.sigmoid_with_loss import SigmoidWithLoss
from chap4.common.embedding_dot import EmbeddingDot

class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):                       # W:输入侧权重 corpus:单词ID列表
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)                   # 负采样 sample_size
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]      # 一个正例+sample_size负例
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 正例的正向传播 假设第一层为正例
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)                         # 正例的正确解标签为 1
        loss = self.loss_layers[0].forward(score, correct_label)

        # 负例的正向传播 假设除第一层外都是负例
        negative_label = np.zeros(batch_size, dtype=np.int32)                       # 负例的正确解标签为 0
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)

        return dh






