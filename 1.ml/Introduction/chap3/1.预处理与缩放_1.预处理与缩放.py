import mglearn
import numpy as np
from matplotlib import pyplot as plt
from mglearn import cm2
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer

if __name__ == '__main__':
    # 1.不同类型的数据预处理
    X, y = make_blobs(n_samples=50, centers=2, random_state=4, cluster_std=1)
    X += 3
    plt.figure(figsize=(15, 8))
    # 构建左边的大子图
    big_ax = plt.subplot2grid((2, 4),          # 创建一个2d子图 2行4列共8个子图
                               (0, 0),            # 子图起始位置0行0列
                               rowspan=2, colspan=2)  #  子图在垂直方向上跨越2行,水平方向上跨越2列 从第0行到第1行 从第0列到第1列
    big_ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cm2, s=60)
    max_x = np.abs(X[:, 0]).max()
    max_y = np.abs(X[:, 1]).max()
    big_ax.set_xlim(-max_x + 1, max_x + 1)
    big_ax.set_ylim(-max_y + 1, max_y + 1)
    big_ax.set_title("Original Data")
    # 构建右边小子图
    other_axes = [plt.subplot2grid((2, 4), (i, j))
                    for i in range(2)
                    for j in range(2, 4)]
    for ax, scaler in zip(other_axes, [StandardScaler(), RobustScaler(), MinMaxScaler(), Normalizer(norm='l2')]):
        X_ = scaler.fit_transform(X)
        ax.scatter(X_[:, 0], X_[:, 1], c=y, cmap=cm2, s=60)
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_title(type(scaler).__name__)

    other_axes.append(big_ax)
    for ax in other_axes:
        ax.spines['left'].set_position('center')    # 左侧的坐标轴线(y轴)移动到图形的中心位置
        ax.spines['right'].set_color('none')        # 右侧的坐标轴线隐藏
        ax.spines['bottom'].set_position('center')  # 底部的坐标轴线(x轴)移动到图形的垂直中心位置
        ax.spines['top'].set_color('none')          # 顶部的坐标轴线隐藏
        ax.xaxis.set_ticks_position('bottom')       # x轴的刻度线仅显示在底部
        ax.yaxis.set_ticks_position('left')         # y轴的刻度线仅显示在左侧
    plt.savefig("./../img/3.无监督学习与预处理/1.对数据集缩放和预处理的各种方法.png",dpi=1080)
    plt.show()

    """
    StandardScaler()
        每个特征的平均值为0,方差为1,保证所有特征都位于同一量级,但不能保证特征任何特定的最大值和最小值
    RobustScaler()
        使用中位数和四分位数确保每个特征的统计属性都位于同一范围  中位数/较小四分位数/较大四分位数
        会忽略与其他点有很大不同的数据点 即忽略异常值
    MinMaxScaler()
        使所有特征都刚好位于0到1之间
    Normalizer(norm='l2')
        对每个数据点进行缩放,使得特征向量的欧式长度等于1 它将一个数据点投射到半径为1的圆(球面)上
        每个数据点的缩放比例都不相同,若只有数据的方向(或角度)是重要的,而特征向量的长度无关紧要,通常使用此归一化方式
    """
