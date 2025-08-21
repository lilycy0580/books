import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    np.random.seed(3)   # 随机种子

    N = 2               # mini-batch的大小
    H = 3               # 隐藏状态向量的维数
    T = 20              # 时序数据的长度 T = 20

    dh = np.ones((N, H))
    # Wh = np.random.randn(H, H)                  # 梯度爆炸
    Wh = np.random.randn(H, H) * 0.5          # 梯度消失

    norm_list = []
    for t in range(T):
        dh = np.dot(dh, Wh.T)
        norm = np.sqrt(np.sum(dh ** 2)) / N     # 将dh的范数添加到list中 mini-batch中的平均"L2范数"(对所有元素的平方和求平方根)
        norm_list.append(norm)

    print(norm_list)

    # 绘制图形
    plt.plot(np.arange(len(norm_list)), norm_list)
    plt.xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
    plt.xlabel('time step')
    plt.ylabel('norm')
    plt.show()
    # plt.savefig('梯度爆炸.png')
    plt.savefig('梯度消失.png')


