
import numpy as np
from chap3.common.layer import MatMul, SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size                      # 词汇的个数,中间层神经元个数
        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')     # 输入侧权重:行方向上排列单词向量
        W_out = 0.01 * np.random.randn(H, V).astype('f')    # 输出侧权重:列方向上排列单词向量  不同于chap4 improved_cbow

        # 生成层
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()

        # 将所有的权重和梯度整理到列表中
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:                                # 此处多个层共享相同的权重,Adam,Momentum等优化器的运行会变得不符合预期
            self.params += layer.params                     # 在更新参数时会进行简单的去重操作 trainer.py的remove_duplicate()
            self.grads += layer.grads

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:, 0])
        h1 = self.in_layer1.forward(contexts[:, 1])
        h = (h0 + h1) * 0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss

    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None

