import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

if __name__ == '__main__':
    """
    DBSCAN        无法创建多于一个较大的簇
    凝聚聚类与k均值 更可能创建均匀大小的簇 需要设置簇的目标个数 
    """
    # Kmeans 先设置一个比较小的簇的数量,比如10个
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

    # 使用Kmeans分析人脸数据集
    km = KMeans(n_clusters=10, random_state=0)
    labels_km = km.fit_predict(X_pca)                                 # k均值聚类将数据划分为大小相似的簇
    print("Cluster sizes k-means: {}".format(np.bincount(labels_km))) # [113 256 188 147 216 180 258 211 139 355]

    fig, axes = plt.subplots(2, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(12, 4))
    for center, ax in zip(km.cluster_centers_, axes.ravel()):
        ax.imshow(pca.inverse_transform(center).reshape(image_shape), cmap='gray', vmin=0, vmax=1)
    plt.savefig("./../img/3.无监督学习与预处理/43.将簇的数量设置为10时,k均值找到的簇中心.png", dpi=1080)
    plt.show()
    # 总结:k均值找到的簇中心是非常平滑的人脸 聚类似乎捕捉到人脸的不同方向,不同表情,以及是否有衬衫领子

    # 将簇中心可视化来进一步分析k均值的结果(使用 pca.inverse_transform 将簇中心旋转回到原始空间并可视化)
    # 每个簇中心给出了簇中5张最典型的图像(该簇中与簇中心距离最近的图像)与5张最不典型的图像(该簇中与簇中心距离最远的图像)
    # mglearn.plots.plot_kmeans_faces(km, pca, X_pca, X_people, y_people, people.target_names)
    n_clusters = 10
    fig, axes = plt.subplots(n_clusters, 11, subplot_kw={'xticks': (), 'yticks': ()},figsize=(10, 15), gridspec_kw={"hspace": .3})
    for cluster in range(n_clusters):
        center = km.cluster_centers_[cluster]           # 第i个簇中心
        mask = km.labels_ == cluster                    # 创建掩码标识属于当前聚类的样本
        dists = np.sum((X_pca - center) ** 2, axis=1)   # 计算所有样本到当前聚类中心的距离 (2063,)

        dists[~mask] = np.inf                           # 在dists数组中对应mask为false的位置设置为无穷大
        data_near = np.argsort(dists)[:5]               # 取距离最近的5个点 (最典型)
        dists[~mask] = -np.inf
        data_far = np.argsort(dists)[-5:]               # 取距离最远的5个点 (最不典型)
        data = np.r_[data_near,data_far]                # 数组拼接 (10,) [1089 1786  811  408   51  279 2036 1308   23 1989]

        rebuild_img = pca.inverse_transform(center).reshape(image_shape)
        axes[cluster, 0].imshow(rebuild_img, cmap='gray', vmin=0, vmax=1)

        for image, label, asdf, ax in zip(X_people[data], y_people[data], km.labels_[data], axes[cluster, 1:]):
            ax.imshow(image.reshape(image_shape), cmap='gray', vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})

    rec = plt.Rectangle([-5, -30], 73, 1295, fill=False, lw=2)  # 绘制矩形
    rec = axes[0, 0].add_patch(rec)                                             # 将矩形添加到子图 axes[0, 0] 中
    rec.set_clip_on(False)                                                      # 允许矩形超出子图边界显示(不裁剪)
    axes[0, 0].text(0, -40, "Center")                                           # 添加文本标签

    rec = plt.Rectangle([-5, -30], 385, 1295, fill=False, lw=2)
    rec = axes[0, 1].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 1].text(0, -40, "Close to center")

    rec = plt.Rectangle([-5, -30], 385, 1295, fill=False, lw=2)
    rec = axes[0, 6].add_patch(rec)
    rec.set_clip_on(False)
    axes[0, 6].text(0, -40, "Far from center")

    plt.savefig("./../img/3.无监督学习与预处理/44.k均值为每个簇找到的样本图像——簇中心在最左边,然后是五个距中心最近的点,然后是五个距该簇距中心最远的点.png", dpi=1080)
    plt.show()
    """
    总结:
        1.证实第3个簇是笑脸的直觉,证实了其他簇中方向的重要性
        2."非典型的"点与簇中心不太相似,而且它们的分配似乎有些随意 =====> k均值对所有数据点进行划分,不像DBSCAN那样具有"噪声"点的概念
        3.利用更多数量的簇,算法可以找到更细微的区别,但添加更多的簇会使得人工检查更加困难
    """