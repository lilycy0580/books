
from chap4.gpu.np import np
from chap4.common.unigram_sampler import UnigramSampler                 # 负采样
from chap4.common.sigmoid_with_loss import SigmoidWithLoss              # 二分类 sigmoid with loss
from chap4.common.embedding_dot import EmbeddingDot,Embedding           # Embedding+Dot
from chap4.common.negative_sampling_loss import NegativeSamplingLoss    # 负采样loss

class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')                 # 输入侧权重:行方向上排列单词向量
        W_out = 0.01 * np.random.randn(V, H).astype('f')                # 输入侧权重:行方向上排列单词向量,因为使用Embedding层

        # 生成层
        self.in_layers = []
        for i in range(2 * window_size):                                # 先创建2*window_size个Embedding层
            layer = Embedding(W_in)                                     # 后创建NegativeSamplingLoss
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)

        # 将所有的权重和梯度整理到列表中
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):                                # 上下文与目标词  都是单词ID形式,不是one-hot形式
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None