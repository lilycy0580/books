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

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None              # 多个RNN层

        self.h, self.dh = None, None    # h:保存调用 forward()时的最后一个RNN层的隐藏状态
                                        # dh:调用backward()时,保存传给前一个块的隐藏状态的梯度
        self.stateful = stateful        # 是否继承隐藏状态

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape              # xs 囊括了T个时序数据,输入向量的维数是 D,batchsize=N
        D, H = Wx.shape

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')         # 输出容器 hs

        if not self.stateful or self.h is None:            # 首次调用forward()时,h所有矩阵元素均为0
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):                                 # T次for循环中,生成RNN层,并将其添加到成员变量layers中
            layer = RNN(*self.params)                      # 计算RNN层各个时刻的隐藏状态,并存放在hs的对应索引(时刻)中
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')        # dxs,传给下游的梯度的"容器"
        dh = 0
        grads = [0, 0, 0]

        for t in reversed(range(T)):                      # 与正向传播相反,调用RNN层的backward(),求得各个时刻的梯度dx,存放在dxs中
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):        # 求各个RNN层的权重梯度的和
                grads[i] += grad

        for i, grad in enumerate(grads):                  # 用最终结果覆盖成员变量 self.grads
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None



