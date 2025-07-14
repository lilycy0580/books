import matplotlib
import  numpy as np
import mglearn
from matplotlib import pyplot as plt, cycler
from mglearn import cm3
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances

"""
聚类(clustering)
    将数据集划分成组,即簇(cluster)用于划分数据,使得一个簇内的数据点非常相似且不同簇内的数据点非常不同
    与分类算法类似,聚类算法为每个数据点分配(或预测)一个数字,表示这个点属于哪个簇
"""
if __name__ == '__main__':
    # 1.k均值聚类
    """
    k均值聚类:
        算法交替执行以下两个步骤:
            1.将每个数据点分配给最近的簇中心,然后将每个簇中心设置为所分配的所有数据点的平均值
            2.如果簇的分配不再发生变化,那么算法结束
            
        指定寻找三个簇,先指定三个随机数据点为簇中心将算法初始化,然后开始迭代算法
            1.每个数据点被分配给距离最近的簇中心
            2.将簇中心修改为所分配点的平均值
            3.重复上一过程两次,第三次迭代后簇中心的数据点保持不变,算法结束
            
        优点:
            比较流行的聚类算法,容易理解和实现,运行较快,可轻松扩展到大数据集
        
        缺点:
            1.依赖随机化,算法的输出依赖随机种子,默认情况下,用10种不同的随机初始化算法运行10次,仅返回最佳结果
            2.对簇形状的假设的约束性较强,徐指定所要寻找的簇的个数(现实世界中可能并不知道该数字)
    """
    mglearn.plots.plot_kmeans_algorithm()
    X, y = make_blobs(random_state=1)   # (100, 2) (100,) 高斯(正态)分布 三分类[0,1,2]

    # with语句设置临时参数
    #   matplotlib.rc_context() 上下文管理器,用于临时修改全局配置
    #   指定修改的配置项为 axes.prop_cycle,控制绘图时自动循环使用的属性 通过属性循环器cycler对象自定义颜色循环 蓝,红,绿
    with matplotlib.rc_context(rc={'axes.prop_cycle': cycler('color', ['#0000aa','#ff2020','#50ff50'])}):
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), subplot_kw={'xticks': (), 'yticks': ()})
        axes = axes.ravel()

        mglearn.discrete_scatter(X[:, 0], X[:, 1], ax=axes[0], markers=['o'], c='w')    # # ax指向第一个子图,w为白色
        axes[0].set_title("Input data")

        init = X[:3, :]
        mglearn.discrete_scatter(X[:, 0], X[:, 1], ax=axes[1], markers=['o'], c='w')
        mglearn.discrete_scatter(init[:, 0], init[:, 1], [0, 1, 2], ax=axes[1], markers=['^'], markeredgewidth=2)
        axes[1].set_title("Initialization")

        # 先根据初始点标记类别后重新计算簇中心,根据新簇中心重新标记类别
        labels = np.argmin(pairwise_distances(init, X), axis=0) # 为X中的每个点找到离它最近的init中心点的索引 (100,) [0 1 2...]
        km_1 = KMeans(n_clusters=3, init=init, max_iter=1, n_init=1).fit(X)
        km_centers_1 = km_1.cluster_centers_    # 聚类后每个簇的中心点(质心)的坐标  (3, 2)
        km_labels_1 = km_1.labels_              # 聚类后每个数据的类别
        mglearn.discrete_scatter(X[:, 0], X[:, 1], labels, markers=['o'], ax=axes[2])
        mglearn.discrete_scatter(init[:, 0], init[:, 1], [0, 1, 2], ax=axes[2], markers=['^'], markeredgewidth=2)
        axes[2].set_title("Assign Points (1)")
        mglearn.discrete_scatter(X[:, 0], X[:, 1], labels, markers=['o'], ax=axes[3])
        mglearn.discrete_scatter(km_centers_1[:, 0], km_centers_1[:, 1], [0, 1, 2], ax=axes[3], markers=['^'], markeredgewidth=2)
        axes[3].set_title("Recompute Centers (1)")
        mglearn.discrete_scatter(X[:, 0], X[:, 1], km_labels_1, markers=['o'], ax=axes[4])
        mglearn.discrete_scatter(km_centers_1[:, 0], km_centers_1[:, 1], [0, 1, 2], ax=axes[4], markers=['^'], markeredgewidth=2)
        axes[4].set_title("Reassign Points (2)")

        # 重新计算簇中心
        km_2 = KMeans(n_clusters=3, init=init, max_iter=2, n_init=1).fit(X)
        km_centers_2 = km_2.cluster_centers_
        km_labels_2 = km_2.labels_
        mglearn.discrete_scatter(X[:, 0], X[:, 1], km_labels_1, markers=['o'], ax=axes[5])
        mglearn.discrete_scatter(km_centers_2[:, 0], km_centers_2[:, 1], [0, 1, 2],ax=axes[5], markers=['^'], markeredgewidth=2)
        axes[5].set_title("Recompute Centers (2)")
        mglearn.discrete_scatter(X[:, 0], X[:, 1], km_labels_2, markers=['o'], ax=axes[6])
        markers = mglearn.discrete_scatter(km_centers_2[:, 0], km_centers_2[:, 1], [0, 1, 2],ax=axes[6], markers=['^'], markeredgewidth=2)
        axes[6].set_title("Reassign Points (3)")

        # 重新计算簇中心
        km_3 = KMeans(n_clusters=3, init=init, max_iter=3, n_init=1).fit(X)
        km_centers_3 = km_3.cluster_centers_
        km_labels_3 = km_3.labels_
        mglearn.discrete_scatter(X[:, 0], X[:, 1], km_labels_2, markers=['o'], ax=axes[7])
        mglearn.discrete_scatter(km_centers_3[:, 0], km_centers_3[:, 1], [0, 1, 2], ax=axes[7], markers=['^'], markeredgewidth=2)
        axes[7].set_title("Recompute Centers (3)")

        axes[8].set_axis_off()                                                          # 关闭第9个子图的坐标轴和背景
        axes[8].legend(markers, ["Cluster 0", "Cluster 1", "Cluster 2"], loc='best')    # 在第9个子图上添加图例legend
    plt.savefig("./../img/3.无监督学习与预处理/23.输入数据与k均值算法的三个步骤.png", dpi=1080)
    plt.show()

    # 绘制决策边界
    mglearn.plots.plot_kmeans_boundaries()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], km_labels_3, markers=['o'])
    mglearn.discrete_scatter(km_centers_3[:, 0], km_centers_3[:, 1],[0, 1, 2], markers=['^'], markeredgewidth=2)
    mglearn.plots.plot_2d_classification(km_3, X, cm=cm3, alpha=.4)
    plt.savefig("./../img/3.无监督学习与预处理/24.k均值算法找到的簇中心和簇边界.png", dpi=1080)
    plt.show()

    # 对上面的数据重新使用KMeans 聚类算法与分类算法相似,每个元素都有一个标签,但并非真实的标签,无任何先验意义
    km = KMeans(n_clusters=3).fit(X)
    print("Cluster memberships:",km.labels_)
    print(km.predict(X))
    mglearn.discrete_scatter(X[:, 0], X[:, 1], km.labels_, markers='o')
    mglearn.discrete_scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], [0, 1, 2], markers='^', markeredgewidth=2)
    plt.savefig("./../img/3.无监督学习与预处理/25.3个簇的k均值算法找到的簇分配和簇中心.png", dpi=1080)
    plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    km_c2 = KMeans(n_clusters=2).fit(X)
    label_c2 = km_c2.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], label_c2, ax=axes[0])
    km_c5 = KMeans(n_clusters=5).fit(X)
    label_c5 = km_c5.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], label_c5, ax=axes[1])
    plt.savefig("./../img/3.无监督学习与预处理/26.使用2个簇(左)和5个簇(右)的k均值算法找到的簇分配.png", dpi=1080)
    plt.show()





