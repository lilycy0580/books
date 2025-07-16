import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # 人脸数据集
    people = fetch_lfw_people(data_home ="./../data", min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255.

    # 从人脸数据集提起特征并进行数据变换
    pca = PCA(n_components=100, whiten=True, random_state=0)
    X_pca = pca.fit_transform(X_people)

    # 使用DBSCAN分析人脸数据集
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels))) # [-1] 所有数据标记为噪声 默认:min_samples=5,eps=0.5

    dbscan = DBSCAN(min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels))) # [-1] 减少min_samples,将更小的点组视为簇

    dbscan = DBSCAN(min_samples=3, eps=15)
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels))) # [-1  0] 单一簇和噪声点 增大eps,扩展每个点的邻域

    # 计算所有簇中的点数和噪声中的点数 31个噪声
    print("Number of points per cluster: {}".format(np.bincount(labels + 1))) # [  31 2032]

    # 查看所有噪声点
    noise = X_people[labels == -1]
    fig, axes = plt.subplots(3, 9, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
    for image, ax in zip(noise, axes.ravel()):
        ax.imshow(image.reshape(image_shape),cmap='gray', vmin=0, vmax=1)
    plt.savefig("./../img/3.无监督学习与预处理/41.人脸数据集中被DBSCAN标记为噪声的样本.png", dpi=1080)
    plt.show()
    # 总结:异常值检测 有人喝水,有人带帽子,有人出现手,其他都是角度奇怪(太近或太远) 均是数据中存在的问题,需解决

    # 获取更有趣的簇,但不是一个非常大的簇  eps↓
    for eps in [1, 3, 5, 7, 9, 11, 13]:
        print("eps={}".format(eps))
        dbscan = DBSCAN(eps=eps, min_samples=3)
        labels = dbscan.fit_predict(X_pca)
        print("Number of clusters: {}".format(len(np.unique(labels))))
        print("Cluster sizes: {}".format(np.bincount(labels + 1)))
    """
    对于较小的eps,所有点都被标记为噪声
    eps=7时,我们得到许多噪声点和许多较小的簇
    eps=9时,我们仍得到许多噪声点,但我们得到了一个较大的簇和一些较小的簇
    eps=11开始,我们仅得到一个较大的簇和噪声

    较大的簇从来没有超过一个,最多有一个较大的簇包含大多数点
        这表示数据中没有两类或三类非常不同的人脸图像,而是所有图像或多或少地都与其他图像具有相同的相似度

    eps=1
    Number of clusters: 1
    Cluster sizes: [2063]

    eps=3
    Number of clusters: 1
    Cluster sizes: [2063]

    eps=5
    Number of clusters: 1
    Cluster sizes: [2063]

    eps=7
    Number of clusters: 14
    Cluster sizes: [2003    4   14    7    4    3    3    4    4    3    3    5    3    3]

    eps=9
    Number of clusters: 4
    Cluster sizes: [1306  751    3    3]

    eps=11
    Number of clusters: 2
    Cluster sizes: [ 413 1650]

    eps=13
    Number of clusters: 2
    Cluster sizes: [ 120 1943]
    """

    # eps=7的结果看起来最有趣,它有许多较小的簇 查看这13个小簇
    dbscan = DBSCAN(min_samples=3, eps=7)
    labels = dbscan.fit_predict(X_pca)  # [-1  0  1  2  3  4  5  6  7  8  9 10 11 12]
    for cluster in range(max(labels) + 1):
        mask = labels == cluster
        n_images = np.sum(mask)
        fig, axes = plt.subplots(1, n_images, figsize=(n_images * 1.5, 4), subplot_kw={'xticks': (), 'yticks': ()})
        for image, label, ax in zip(X_people[mask], y_people[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1,cmap='gray')
            ax.set_title(people.target_names[label].split()[-1])
        plt.savefig("./../img/3.无监督学习与预处理/42.eps=7的DBSCAN找到的簇-"+str(cluster)+".png", dpi=1080)
    plt.show()
    """
    总结:
        有一些簇对应于脸部非常不同的人 sharon与Koizumi
        每个簇内人脸方向和面部表情也是固定的
        有的簇中包含多个人的面孔,但他们的方向和表情相似  
    
    此次使用人工分析,不同于监督学习的基于R^2分数或精度的更为自动化的搜索方法
    """
