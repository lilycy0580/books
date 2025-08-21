import os
import pickle
import sys
import numpy as np
from mnist.mnist import load_mnist

# 1.获取数据
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False) # 标准化处理
    return x_test, t_test

# 2.初始化权重
def init_network():
    with open("./mnist/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 3.前向传播
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp = np.exp(a-c)       # 防溢出
    sum_exp = np.sum(exp)
    y = exp/sum_exp
    return y

if __name__ == '__main__':
    """
        mnist数据集 6wdata,1wtest,28*28,像素值[0,255]
        pickle:将程序运行中的对象保存为文件
        数据预处理:
            1.利用数据整体的均值或标准差,移动数据,使数据整体以0为中心分布
            2.对数据进行标准化处理,将数据的延展控制在一定范围内
            3.将数据整体的分布形状均匀化 即数据白化
    """

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    print(x_train.shape, t_train.shape, x_test.shape, t_test.shape) # (60000, 784) (60000,) (10000, 784) (10000,)

    # 神经网络的推理处理 input:img 784个神经元   output:label 10个神经元
    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p= np.argmax(y) # 获取概率最高的元素的索引
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352

    # 神经网络的推理处理——批处理
    x, t = get_data()
    network = init_network()
    batch_size = 100  # 批数量
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i: i+batch_size]
        y_batch = predict(network, x_batch)
        temp_max = np.argmax(y_batch, axis=1)               # axis=1表示行,axis=0表示列
        accuracy_cnt += np.sum(temp_max == t[i: i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))  # Accuracy:0.9352





