import numpy as np
from matplotlib import pyplot as plt
from mglearn import cm3
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    """
    使用聚类算法时,通常没有真实值来比较结果.
        若知道数据的正确聚类,则可以使用此信息构建一个监督模型(如分类器),使用类似ARI和NMI的指标通常有助于开发算法,但对评估应用是否成功没有帮助
    
    评估指标不需要真实值:
        轮廓系数 计算一个簇的紧致度,值越大越好,最大分数为1,但不允许复杂的形状 但在实践中效果不好 
    
    对于评估聚类,稍好的策略是使用基于鲁棒性的聚类指标 sklearn未实现
        算法:
            先向数据中添加一些噪声,或者使用不同的参数设定,然后运行算法,并对结果进行比较
        思想:
            如果许多算法参数和许多数据扰动返回相同的结果,那么它很可能是可信的
    """
    # 轮廓分数 在two_moons数据集 比较Kmeans/凝聚聚类/DBSCAN三种算法
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    X_scaled = StandardScaler().fit(X).transform(X)
    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))
    sc_score = silhouette_score(X_scaled, random_clusters)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c = random_clusters, cmap=cm3, s=60)
    axes[0].set_title("Random assignment:{:.2f}".format(sc_score))

    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2),DBSCAN()]
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        sc_score = silhouette_score(X_scaled, clusters)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=cm3, s=60)
        ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__,sc_score))
    plt.savefig("./../img/3.无监督学习与预处理/40.利用无监督的轮廓分数在two_moons数据集上比较随机分配,k均值,凝聚聚类和DBSCAN(更符合直觉的DBSCAN的轮廓分数低于k均值找到的分配).png", dpi=1080)
    plt.show()
    # 总结:Kmeans轮廓分数最高,尽管我们喜欢DBSCAN的结果
