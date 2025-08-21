import numpy as np
from matplotlib import pyplot as plt
from chap4.mnist.mnist import load_mnist

# 均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

# 交叉熵误差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))       # 避免np.log(0)无穷大

# mini-batch交叉熵误差  one-hot表示
def cross_entropy_error_onehot(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# mini-batch交叉熵误差  非one-hot表示
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
"""
batch_size = 5:
    np.arange(batch_size)           [0, 1, 2, 3, 4]
    t                               [2, 7, 0, 9, 4]
    y[np.arange(batch_size),t]      [y[0,2], y[1,7], y[2,0], y[3,9], y[4,4]]
"""


if __name__ == '__main__':
    # 1.针对单个数据的损失函数
    # 均方误差 计算神经网络的输出和正确解训练数据的各个元素之差的平方的和
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]    # 神经网络输出值 yk
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]                          # 真实标签      tk

    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]                          # 确定"2"为正确解
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]    # 神经网络输出 "2"  损失函数(均方差误差)值更小,与训练数据更吻合
    mse1 = mean_squared_error(np.array(y), np.array(t))
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]   # 神经网络输出 "7"
    mse2 = mean_squared_error(np.array(y), np.array(t))
    print(mse1, mse2)                                           # 0.09750000000000003 0.5975

    # 交叉熵误差 交叉熵误差的值是由正确解标签所对应的输出结果决定的
    cee1 = -np.log(0.6) # 0.5108256237659907
    cee2 = -np.log(0.1) # 2.3025850929940455

    # 交叉熵误差的曲线  神经网络的输出yk值越大,则loss越小
    x = np.arange(0, 1.1, 0.1)
    y = -np.log(x)
    plt.plot(x, y, label='-logx')
    plt.legend()
    plt.savefig('./img/1.png')
    plt.show()

    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    cee1 = cross_entropy_error_onehot(np.array(y), np.array(t))
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
    cee2 = cross_entropy_error_onehot(np.array(y), np.array(t))
    print(cee1, cee2)

    # 2.mini-batch学习
    # 从训练数据中随机选择指定个数的数据,进行mini-batch学习
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape, x_test.shape, t_train.shape, t_test.shape) # (60000, 784) (10000, 784) (60000, 10) (10000, 10)

    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)   # 随机从6w的训练数据中抽取10个样本数据  使用掩码,筛选数据
    x_batch = x_train[batch_mask]                   # (10, 784)
    t_batch = t_train[batch_mask]                   # (10, 10)
    print(batch_mask, x_batch.shape, t_batch.shape) # [43345 30786  3980 36027 53578 46245 27968 20814 13121 14477]





