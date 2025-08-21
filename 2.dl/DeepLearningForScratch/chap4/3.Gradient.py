import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient_net

def f2(x):
    return x[0]**2 + x[1]**2

# 梯度
def numerical_gradient(f, x):
     h = 1e-4                       # 0.0001 避免无穷大
     grad = np.zeros_like(x)        # 生成和x形状相同的数组,元素全为0
     for index in range(x.size):
         tmp = x[index]
         x[index] = tmp + h         # f(x+h)的计算
         fx_h1 = f(x)
         x[index] = tmp - h         # f(x-h)的计算
         fx_h2 = f(x)
         grad[index] = (fx_h1 - fx_h2) / (2*h)
         x[index] = tmp # 还原值
     return grad

# 学习率  求f的极小值
def gradient_descent(f, init_x, lr=0.01, step_num=100):
     x = init_x
     for i in range(step_num):
         grad = numerical_gradient(f, x)
         x -= lr * grad
     return x

# 简单的神经网络
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

# 神经网络的梯度
def f(W):
    return net.loss(x, t)

if __name__ == '__main__':
    # 梯度
    gradient1 = numerical_gradient(f2, np.array([3.0, 4.0]))    # [6. 8.] (2,)
    gradient2 = numerical_gradient(f2, np.array([0.0, 2.0]))    # [0. 4.] (2,)
    gradient3 = numerical_gradient(f2, np.array([3.0, 0.0]))    # [6. 0.] (2,)
    print("梯度:\n",gradient1,gradient1.shape,gradient2,gradient2.shape,gradient3,gradient3.shape)

    # 学习率
    init_x = np.array([-3.0, 4.0])
    descent = gradient_descent(f2, init_x=init_x, lr=0.1, step_num=100) # (-6.1e-10, 8.1e-10)接近(0,0)
    print("学习率:\n",descent)

    # 神经网络的梯度  一次梯度下降权重参数迭代过程
    net = simpleNet()
    print("神经网络的初始权重:\n",net.W)

    x = np.array([0.6, 0.9])
    p = net.predict(x)
    predict_label = np.argmax(p)  # 最大值的索引
    print("神经网络的预测值与预测标签:\n",p,predict_label)

    t = np.array([0, 0, 1]) # 正确解标签
    net.loss(x, t)
    dW = numerical_gradient_net(f, net.W)
    print("神经网络一次梯度下降后的权重:\n",dW)

    # 根据损失函数,求解神经网络的梯度  f一般为损失函数
    f = lambda w: net.loss(x, t)        # 定义匿名函数,将其赋值给变量f  接收参数w,返回
    dW = numerical_gradient(f, net.W)   # 求出神经网络的梯度后，接下来只需根据梯度法，更新权重参数即可

