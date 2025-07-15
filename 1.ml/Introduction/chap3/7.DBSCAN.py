import mglearn
import numpy as np
from matplotlib import pyplot as plt
from mglearn import cm3
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    """
    DBSCAN:
        具有噪声的基于密度的空间聚类应用 
    
    原理:
        簇形成数据的密集区域,并由相对较空的区域分隔开(将彼此距离小于eps的核心样本放到同一个簇中)
        密集区域:识别特征空间的"拥挤"区域中的点即
        核心样本:在密集区域内的点 
                如果在距一个给定数据点eps的距离内数据点>=min_samples个,则该数据点为核心样本
        核心点
        边界点 与核心点距离eps之内的点
        噪声   
       
    算法:
        任意选取一个点,然后找到与该点的距离<=eps的所有点:
            若点的总个数<min_samples,则该点标记为噪声,即不属于任何簇
            若点的总个数>=min_samples,则该点标记为核心样本,并被分配一个新的簇标签
                随后访问该点的所有邻居(距离<=eps),若没有被分配簇,则将刚刚的新簇分配给他们,若他们是核心样本,则依次访问其邻居,以此类推
            簇逐渐增大,直到在簇的eps距离内没有更多的核心样本为止 
        选取另一个尚未被访问过的点,重复上述过程
        
    优点:
        不需要用户先验地设置簇的个数,可以划分具有复杂形状的簇,还可以找出不属于任何簇的点
        DBSCAN比凝聚聚类和k均值稍慢,但仍可以扩展到相对较大的数据集
    
    缺点:
        同凝聚聚类,DBSCAN不允许对新的测试数据进行预测 
    """

    X, y = make_blobs(random_state=0, n_samples=12)
    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X)
    print("Cluster memberships:",clusters) # 参数取默认值,所有数据点都被分配了标签-1,表示噪声  [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]

    # min_samples和eps取不同值时的簇分类
    # mglearn.plots.plot_dbscan()
    X, y = make_blobs(random_state=0, n_samples=12)
    fig, axes = plt.subplots(3, 4, figsize=(11, 8), subplot_kw={'xticks': (), 'yticks': ()})
    colors = [cm3(1), cm3(0), cm3(2)] # clusters: red, green and blue    outliers:white
    markers = ['o', '^', 'v']
    for i, min_samples in enumerate([2, 3, 5]):
        for j, eps in enumerate([1, 1.5, 2, 3]):
            dbscan = DBSCAN(min_samples=min_samples, eps=eps)
            clusters = dbscan.fit_predict(X)
            print("min_samples:%d       eps:%f      cluster:%s" % (min_samples, eps, clusters))
            if np.any(clusters == -1):
                c = ['w'] + colors
                m = ['o'] + markers     # 噪声:白点
            else:
                c = colors              # 核心点+边缘点:彩点
                m = markers
            mglearn.discrete_scatter(X[:, 0], X[:, 1], clusters, ax=axes[i, j], c=c, s=8, markers=m)
            inds = dbscan.core_sample_indices_
            if len(inds):
                mglearn.discrete_scatter(X[inds, 0], X[inds, 1], clusters[inds],ax=axes[i, j], s=15, c=colors, markers=markers)
            axes[i, j].set_title("min_samples:%d    eps:%.1f" % (min_samples, eps))
    fig.tight_layout()
    plt.savefig("./../img/3.无监督学习与预处理/37.在min_samples和eps参数不同取值的情况下,DBSCAN找到的簇分配.png", dpi=1080)
    plt.show()
    """
    属于簇的点为实心的,噪声点为空心
    核心样本为较大的标记,边界点为较小的标记
    min_samples:2       eps:1.000000      cluster:[-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
    min_samples:2       eps:1.500000      cluster:[0 1 1 1 1 0 2 2 1 2 2 0]
    min_samples:2       eps:2.000000      cluster:[0 1 1 1 1 0 0 0 1 0 0 0]
    min_samples:2       eps:3.000000      cluster:[0 0 0 0 0 0 0 0 0 0 0 0]
    min_samples:3       eps:1.000000      cluster:[-1  0  0 -1  0 -1  1  1  0  1 -1 -1]
    min_samples:3       eps:1.500000      cluster:[0 1 1 1 1 0 2 2 1 2 2 0]
    min_samples:3       eps:2.000000      cluster:[0 1 1 1 1 0 0 0 1 0 0 0]
    min_samples:3       eps:3.000000      cluster:[0 0 0 0 0 0 0 0 0 0 0 0]
    min_samples:5       eps:1.000000      cluster:[-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]
    min_samples:5       eps:1.500000      cluster:[-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
    min_samples:5       eps:2.000000      cluster:[-1  0  0  0  0 -1 -1 -1  0 -1 -1 -1]
    min_samples:5       eps:3.000000      cluster:[0 0 0 0 0 0 0 0 0 0 0 0]
    
    增大eps: 更多的点会被包含在一个簇中,让簇变大,但也导致多个簇合并成一个
    增大min_samples:核心点会变少,更多的点标记为噪声
    
    eps:
        决定了点与点之间"接近",
            设置较小,则没有核心样本,导致所有点标记为噪声 
            设置过大导致所有点形成单个簇
        
    min_samples:
        判断稀疏区域内的点被标记为异常值还是形成自己的簇,决定簇的最小尺寸
            设置过大任何一个包含少于min_samples个样本的簇现在将被标记为噪声
        
        DBSCAN不需要显示设置簇的个数,但设置eps可以隐式地控制找到的簇的个数
        通过StandardScaler/MinMaxScaler对数据进行缩放后,更容易找出eps的较好的值
    """
    # DBSCAN two_moons
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)
    X_scaled = StandardScaler().fit(X).transform(X)
    clusters = DBSCAN().fit_predict(X_scaled)   # (200,) [0 1 1...]
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/38.利用默认值eps=0.5的DBSCAN找到的簇分配.png", dpi=1080)
    plt.show()