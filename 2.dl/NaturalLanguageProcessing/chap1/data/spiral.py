# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100                 # 各类的样本数
    dim = 2                 # 数据的元素个数
    class_num = 3           # 类别数

    x = np.zeros((N*class_num, dim))
    t = np.zeros((N*class_num, class_num), dtype=np.int32)

    for j in range(class_num):
        for i in range(N):  #N*j, N*(j+1)):
            rate = i / N
            radius = 1.0*rate
            theta = j*4.0 + 4.0*rate + np.random.randn()*0.2
            ix = N*j + i
            x[ix] = np.array([radius*np.sin(theta),radius*np.cos(theta)]).flatten()
            t[ix, j] = 1
    return x, t

if __name__ == '__main__':
    x, t = load_data()
    print('x', x.shape)  # (300, 2)
    print('t', t.shape)  # (300, 3)

    # 绘制数据点
    N = 100
    class_num = 3
    markers = ['o', 'x', '^']
    for i in range(class_num):
        plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
    plt.savefig('./spiral.png' )
    plt.show()