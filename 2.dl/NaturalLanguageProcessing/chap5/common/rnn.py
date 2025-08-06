
from chap5.gpu.np import np


class RNN:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]   # 接收两个权重参数和一个偏置参数
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None           # 反向传播时要用到的中间数据 cache 进行初始化

    def forward(self, x, h_prev):   # 正向传播接收2个参数:从下方输入的 x, 从左边输入的 h_prev
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)
        self.cache = (x, h_prev, h_next)
        return h_next               # 当前时刻的 RNN 层的输出 = 下一时刻的 RNN 层的输入

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next ** 2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev