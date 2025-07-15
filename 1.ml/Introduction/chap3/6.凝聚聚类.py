
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity
from scipy.cluster.hierarchy import dendrogram, ward

if __name__ == '__main__':
    """
    凝聚聚类:
        许多基于相同原则构建的聚类算法,算法首先声明每个点是自己的簇,然后合并两个最相似的簇,直到满足某种停止准则为止

    sklearn中算法:
        停止准则:指定簇的个数
        链接准则:规定如何度量"最相似的簇"
            ward:挑选两个簇来合并使得所有簇中的方差增加最小
            average:将簇中所有点之间平均距离最小的两个簇合并
            complete:将簇中点之间最大距离最小的两个簇合并 即最大链接

            ward适用于大多数数据集
            如果簇中的成员个数非常不同(如其中一个比其他所有都大得多),则average或complete可能效果更好
            
    聚类算法无法分离向two_moons这样形状复杂的数据集,但是DBSCAN可以
    """
    # mglearn.plots.plot_agglomerative_algorithm()
    X, y = make_blobs(random_state=0, n_samples=12)
    agg = AgglomerativeClustering(n_clusters=X.shape[0],    # 设置最终聚类数量等于样本数
                                  compute_full_tree=True)   # 强制计算完整的树结构,记录所有合并步骤
    fig, axes = plt.subplots(X.shape[0] // 5, 5, subplot_kw={'xticks': (),'yticks': ()}, figsize=(20, 8)) # 绘制12子图,展示每一步聚类结果
    eps = X.std() / 2
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps # 指定x轴/y轴范围,并扩展半个标准差避免边缘截断 eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    x_range = np.linspace(x_min, x_max, 100)           # (100,)
    y_range = np.linspace(y_min, y_max, 100)
    xx, yy = np.meshgrid(x_range, y_range)              # (100, 100)
    gridpoints = np.c_[xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)] # (10000, 2) 将两个一维数组转换为二维网格坐标矩阵
    for i, ax in enumerate(axes.ravel()):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title("Step %d" % i)
        ax.scatter(X[:, 0], X[:, 1], s=60, c='grey')
        agg.n_clusters = X.shape[0] - i
        agg.fit(X)                      # 对X执行层次聚类,生产聚类标签
        count = np.bincount(agg.labels_) # (12,) [1 1 1 1 1 1 1 1 1 1 1 1] .... (3,) [4 5 3]
        # 给数据点>2的各个簇绘制分界线
        for cluster in range(agg.n_clusters):
            if count[cluster] > 1:
                points = X[agg.labels_ == cluster]              # 筛选出特定簇的数据点
                other_points = X[agg.labels_ != cluster]
                kde = KernelDensity(bandwidth=.5).fit(points)   # 创建高斯核密度模型,带宽0.5 使用points数据拟合模型,计算概率密度分布
                scores = kde.score_samples(gridpoints)          # 返回每个网格点的对数概率密度值
                score_inside = np.min(kde.score_samples(points))# 原始数据点中概率密度最低的值
                score_outside = np.max(kde.score_samples(other_points))# 外部数据点(可能是背景或噪声)中概率密度最高的值
                levels = .8 * score_inside + .2 * score_outside # 将阈值设为score_inside和score_outside的加权平均(80%边缘,20%外部),目的是在两者之间绘制等高线
                ax.contour(xx, yy, scores.reshape(100, 100), levels=[levels], colors='k', linestyles='solid', linewidths=2)
    axes[0, 0].set_title("Initialization")
    plt.savefig("./../img/3.无监督学习与预处理/33.凝聚聚类用迭代的方式合并两个最近的簇.png", dpi=1080)
    plt.show()
    """
    总结:
        step0.最开始每个点自成一簇.
        step1-4.相距最近的两个簇被合并,选出两个单点簇并将其合并成两点簇
        step5-8.其中一个两点簇被扩展到三个点,以此类推
        step9.只剩下3个簇,算法结束
    """

    # 凝聚聚类 对三点簇数据进行聚类 效果完美
    X, y = make_blobs(random_state=1)
    agg = AgglomerativeClustering(n_clusters=3)
    predict = agg.fit_predict(X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], predict)
    plt.legend(["Cluster 0", "Cluster 1", "Cluster 2"], loc="best")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/34.使用3个簇的凝聚聚类的簇分配.png", dpi=1080)
    plt.show()

    """
    层次聚类与树状图
        凝聚聚类生成了层次聚类,聚类过程迭代进行,每个点都从一个单点簇变为属于最终的某个簇,每个中间步骤都提供了数据的一种聚类
            二维数据 sklearn
            多维数据 Scipy 特征>2个 树状图 
    """
    # 二维
    # mglearn.plots.plot_agglomerative()
    X, y = make_blobs(random_state=0, n_samples=12)
    eps = X.std() / 2.
    x_min, x_max = X[:, 0].min() - eps, X[:, 0].max() + eps
    y_min, y_max = X[:, 1].min() - eps, X[:, 1].max() + eps
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    gridpoints = np.c_[xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)]
    ax = plt.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    ax.scatter(X[:, 0], X[:, 1], s=60, c='grey')
    # 在(x[0] + 0.1, x[1])处添加文本标签   文本内容为 "%d" % i, 将索引i格式化为整数显示
    for i, x in enumerate(X):
        ax.text(x[0] + 0.1, x[1], "%d" % i, horizontalalignment='left', verticalalignment='center')
    agg = AgglomerativeClustering(n_clusters=3)
    for i in range(11):
        agg.n_clusters = X.shape[0] - i
        agg.fit(X)
        bins = np.bincount(agg.labels_)
        for cluster in range(agg.n_clusters):
            if bins[cluster] > 1:
                points = X[agg.labels_ == cluster]
                other_points = X[agg.labels_ != cluster]
                kde = KernelDensity(bandwidth=.5).fit(points)
                scores = kde.score_samples(gridpoints)
                score_inside = np.min(kde.score_samples(points))
                score_outside = np.max(kde.score_samples(other_points))
                levels = .8 * score_inside + .2 * score_outside
                ax.contour(xx, yy, scores.reshape(100, 100), levels=[levels],colors='k', linestyles='solid', linewidths=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    plt.tight_layout()
    plt.savefig("./../img/3.无监督学习与预处理/35.凝聚聚类生成的层次化的簇分配(用线表示)以及带有编号的数据点(参见图3-36).png", dpi=1080)
    plt.show()

    # 多维
    X, y = make_blobs(random_state=0, n_samples=12)
    linkage_array = ward(X)
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')
    ax.text(bounds[1], 7.25, ' two clusters', va='center', fontdict={'size': 15})
    ax.text(bounds[1], 4, ' three clusters', va='center', fontdict={'size': 15})
    plt.xlabel("Sample index")
    plt.ylabel("Cluster distance")
    plt.tight_layout()
    plt.savefig("./../img/3.无监督学习与预处理/36.聚类的树状图(用线表示划分成两个簇和三个簇).png", dpi=1080)
    plt.show()
    # y轴表示被合并的簇之间的距离 从三个簇到两个簇的过程中合并了一些距离非常远的点 / 将剩下的两个簇合并为一个簇也需要跨越相对较大的距离


