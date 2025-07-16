import mglearn
import numpy as np
from matplotlib import pyplot as plt
from mglearn import cm3
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    """
    评估指标:
        评估聚类算法相对于真实聚类的结果,给出定量的度量,其最佳值为1,0表示不相关的聚类
            ARI 调整rand指数
            NMI 归一化互信息

        注意:
            不要使用accuracy_score(准确率)作为评估指标
                它要求分配的簇标签与真实值完全匹配,但簇标签本身毫无意义——唯一重要的是哪些点位于同一个簇中
    """
    # ARI Kmeans 凝聚聚类 DBSCAN
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    X_scaled = StandardScaler().fit(X).transform(X)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})

    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X)) # 均匀分布生产随机整数,用于随机初始化每个样本的簇标签
    score_ARI =  adjusted_rand_score(y, random_clusters)
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c = random_clusters,cmap= cm3, s=60)
    axes[0].set_title("Random assignment - ARI:{:.2f}".format(score_ARI))

    algorithms = [KMeans(n_clusters=2),AgglomerativeClustering(n_clusters=2),DBSCAN()]
    for ax, algorithm in zip(axes[1:], algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c = clusters, cmap=mglearn.cm3, s=60)
        score_ARI = adjusted_rand_score(y, random_clusters)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,score_ARI))
    plt.savefig("./../img/3.无监督学习与预处理/39.利用监督ARI分数在two_moons数据集上比较随机分配,k均值,凝聚聚类和DBSCAN.png", dpi=1080)
    plt.show()
    # score_ARI: 随机簇分数为0,DBSCAN分数为1

    # 注意不要使用准确率
    clusters1 = [0, 0, 1, 1, 0]
    clusters2 = [1, 1, 0, 0, 1]
    accuracy_score = accuracy_score(clusters1, clusters2)
    score_ARI = adjusted_rand_score(clusters1, clusters2)
    print("Accuracy: {:.2f}".format(accuracy_score))    # 0.00, 二者标签完全不同
    print("ARI: {:.2f}".format(score_ARI))              # 1.00, 二者聚类完全相同