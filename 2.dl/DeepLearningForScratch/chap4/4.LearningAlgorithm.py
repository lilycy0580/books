from matplotlib import pyplot as plt

from common.functions import *
from common.gradient import numerical_gradient_net
from mnist.mnist import load_mnist

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y

    # x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

     # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient_net(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient_net(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient_net(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient_net(loss_W, self.params['b2'])
        return grads

if __name__ == '__main__':
    """
    神经网络的学习步骤:
        神经网络存在合适的权重和偏置,调整权重和偏置以便拟合训练数据的过程称为"学习"
    
    随机梯度下降法:(SGD)
        step1:从训练数据中随机选出一部分数据,这部分数据称为mini-batch,目标是减小mini-batch的损失函数的值
        step2:求出各个权重参数的梯度,梯度表示损失函数的值减小最多的方向
        step3:将权重参数沿梯度方向进行微小更新
        step4:重复step1-3
    """
    # # 实现手写数字识别的神经网络  2层神经网络
    # net = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)
    # print(net.params['W1'].shape, net.params['b1'].shape, net.params['W2'].shape, net.params['b2'].shape)
    # # (784, 100) (100,) (100, 10) (10,)
    #
    # x = np.random.rand(100, 784)    # 伪输入数据   mini_bath = 100
    # y = net.predict(x)
    # t = np.random.rand(100, 10)     # 伪正确解标签
    # grads = net.numerical_gradient(x, t)  # 计算梯度

    # 从训练数据中随机选择一部分数据,使用梯度法更新参数
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)

    train_loss_list = []

    # 超参数
    iters_num = 1000
    batch_size = 100
    learning_rate = 0.1
    train_size = x_train.shape[0]

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    print(network)

    for i in range(iters_num):
         batch_mask = np.random.choice(train_size, batch_size)  # 获取mini-batch
         x_batch = x_train[batch_mask]
         t_batch = t_train[batch_mask]
         grad = network.numerical_gradient(x_batch, t_batch)     # 计算梯度
         # grad = gradient.gradient(x_batch, t_batch) # 高速版!
         for key in ('W1', 'b1', 'W2', 'b2'):                    # 更新参数
            network.params[key] -= learning_rate * grad[key]     # 记录学习过程
         loss = network.loss(x_batch, t_batch)
         train_loss_list.append(loss)
         if(i % 100 == 0):
            print("第"+str(i)+"次迭代后loss的值:"+str(loss))

    plt.plot(iters_num,train_loss_list)
    plt.show()
    print("done!")
    print(train_loss_list)
    # 损失函数的值在不断减小,神经网络正在逐渐向最优参数靠近

