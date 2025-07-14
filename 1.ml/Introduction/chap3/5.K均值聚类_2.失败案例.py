import mglearn
import numpy as np
from matplotlib import pyplot as plt
from mglearn import cm2
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons

if __name__ == '__main__':
    """
    k均值失败:
        1.即使知道给定数据集中簇的"正确"个数,k均值可能也不是总能找到它们 
            1.每个簇由其中心定义,每个簇都为凸形 k均值只能找到相对简单的形状
            2.k均值还假设所有簇某种程度上具有相同的直径,总是将簇之间的边界刚好画在簇中心的中间位置
            
        2.k均值还假设所有方向对每个簇都同等重要
            k均值仅考虑到最近簇中心的距离,所以它无法处理这种类型的数据    
            如果簇的形状更加复杂,k均值的表现也很差
    """
    # 案例一:簇0和簇1都包含一些远离簇中其他点的点
    X_varied, y_varied = make_blobs(n_samples=200,
                                    cluster_std=[1.0, 2.5, 0.5], # 指定每个簇的标准差 1.0(较紧凑) 2.5(较分散) 0.5(非常紧凑)
                                    random_state=170)
    km = KMeans(n_clusters=3, random_state=0)
    km.fit(X_varied)
    y_pred = km.predict(X_varied)
    mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
    mglearn.discrete_scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], [0,1,2], markers='H', markeredgewidth=3)
    plt.legend(["cluster 0", "cluster 1", "cluster 2"], loc='best')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/27.簇的密度不同时,k均值找到的簇分配.png", dpi=1080)
    plt.show()

    # 案例二:三部分被沿着对角线方向拉长.由于k均值仅考虑到最近簇中心的距离,所以它无法处理这种类型的数据
    X, y = make_blobs(n_samples=600, random_state=170)
    random_state = np.random.RandomState(74)
    transformation = random_state.normal(size=(2, 2))
    X = np.dot(X, transformation)
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='D', markeredgewidth=2)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/28.k均值无法识别非球形簇.png", dpi=1080)
    plt.show()

    # 案例三:希望聚类算法能够发现两个半月形,但是k均值算法做不到
    X, y = make_moons(n_samples=200,    # (200, 2)
                      noise=0.05,       # 控制数据点的分散程度(高斯噪声标准差) 值越大,数据点越分散,分类任务越难(默认0.1)
                      random_state=0)   # 生产半月形/双月形的二维数据集,每个半圆代表一个类别
    km = KMeans(n_clusters=2).fit(X)
    y_pred = km.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cm2, s=60, edgecolor='k')
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], marker='^', c=[cm2(0), cm2(1)], s=100, linewidth=2,edgecolor='k')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/29.k均值无法识别具有复杂形状的簇.png", dpi=1080)
    plt.show()
