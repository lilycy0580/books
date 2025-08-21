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
    # 一个epoch表示学习中所有训练数据均被使用过一次时的更新次数 每经过一个epoch,就对所有的训练数据和测试数据计算识别精度并记录
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label = True)
    train_size = x_train[0].size

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 超参数
    iters_num = 10000
    batch_size = 100
    learning_rate = 0.1

    # 平均每个epoch的重复次数
    iter_per_epoch = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    for i in range(iters_num):
        print(i)
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
        # 计算梯度
        grad = network.numerical_gradient(x_batch, t_batch)
        # grad = network.gradient(x_batch, t_batch) # 高速版!
        # 更新参数
        for key in ('W1', 'b1', 'W2', 'b2'):
            network.params[key] -= learning_rate * grad[key]
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)
        # 计算每个epoch的识别精度
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    plt.plot(iter_per_epoch,train_loss_list)
    plt.show()
    print("done!")

    print(train_acc_list)
    print(test_acc_list)