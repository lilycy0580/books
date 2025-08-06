
from chap5.gpu.np import *


def scatter_add(array, indices, values):
    for i, idx in enumerate(indices):
        array[idx] += values[i]
    return array

class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None                     # 以数组的形式保存需要提取的行的索引(单词ID)

    def forward(self, idx):                 # Embedding层的正向传播从权重矩阵Win中提取特定的行,并将该特定行的神经元原样传给下一层
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):               # Embedding层的反向传播从上一层(输出侧的层)传过来的梯度将原样传给下一层(输入侧的层)
        dW, = self.grads                    # 从上一层传来的梯度会被应用到权重梯度 dW 的特定行,需注意索引重复 不能直接普通加法
        dW[...] = 0
        if GPU:
            # np.scatter_add(dW, self.idx, dout)
            scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)       # 在指定位置进行原位加法,dout的值按照self.idx指定的索引位置,累加到dW数组中
        return None