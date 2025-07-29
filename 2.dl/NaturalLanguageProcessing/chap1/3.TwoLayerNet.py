import numpy as np

# Sigmoid层
class Sigmoid:
    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

# Affine层
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # 生成层
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # 将所有的权重整理到列表中
        self.params = []
        for layer in self.layers:
            self.params += layer.params     # list拼接

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

if __name__ == '__main__':
    # 2.层的类化及正向传播的实现
    """
    所有层都有forward()与backward(),params与grads   

    正向传播:forward()          从输入层到输出层的传播,进行神经网络的学习
        全连接层:Affine层    
        sigmoid函数:Sigmoid层  

    反向传播:backward()         与正向传播相反的顺序传播数据(梯度)
    """
    x = np.random.randn(10, 2)      # 标准正态分布的随机数数组
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)            # (10, 3)
    print(s)
    """
        [[ 0.82685601 -0.2479822  -1.19171514]
         [ 0.80150053  0.05446467 -0.94073349]
         [ 0.8144993   0.21043294 -0.78190241]
         [ 0.81473673  0.90956513 -0.34006619]
         [ 0.72733182  0.48969487 -0.92233189]
         [ 0.83703069  0.21283219 -0.69713458]
         [ 0.94796119 -0.30939755 -0.93254698]
         [ 0.78655948 -0.33589316 -1.34965383]
         [ 0.79387045 -0.04390554 -1.05151703]
         [ 0.84668096  0.59555988 -0.42221335]]    
    """



