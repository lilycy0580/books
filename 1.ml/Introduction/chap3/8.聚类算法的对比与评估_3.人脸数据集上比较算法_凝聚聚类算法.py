import mglearn
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score

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

    # 使用凝聚聚类算法分析人脸数据集
    agglomerative = AgglomerativeClustering(n_clusters=10)
    labels_agg = agglomerative.fit_predict(X_pca)
    print("cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))
    # 总结:凝聚聚类也是生成大小相近的簇,没有Kmeans均匀但比DBSCAN均匀  [478 254 317 119  96 191 424  17  55 112]

    km = KMeans(n_clusters=10, random_state=0)
    labels_km = km.fit_predict(X_pca)

    # ARI来度量凝聚聚类和k均值给出的两种数据划分是否相似
    ari_score = adjusted_rand_score(labels_agg, labels_km)
    print("ARI: {:.2f}".format(ari_score)) # ARI: 0.07 两种聚类算法共同点很少 对于k均值,远离簇中心的点似乎没有什么共同点

    # 绘制层次聚类的树状图,展示数据点如何逐步合并成簇
    linkage_array = ward(X_pca)
    plt.figure(figsize=(20, 8))
    dendrogram(linkage_array, p=7, truncate_mode='level', no_labels=True) # 树状图
    plt.xlabel("Sample index")
    plt.ylabel("Cluster distance")
    plt.tight_layout()
    plt.savefig("./../img/3.无监督学习与预处理/45.凝聚聚类在人脸数据集上的树状图.png", dpi=1080)
    plt.show()

    n_clusters = 10
    for cluster in range(n_clusters):
        mask = labels_agg == cluster
        fig, axes = plt.subplots(1, 10, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(15, 8))
        axes[0].set_ylabel(np.sum(mask))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
            ax.imshow(image.reshape(image_shape), vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1], fontdict={'fontsize': 9})
    plt.savefig("./../img/3.无监督学习与预处理/46.生成的簇中的随机图像——每一行对应一个簇,左侧的数字表示每个簇中图像的数量.png", dpi=1080)
    plt.show()
    # 总结:虽然某些簇似乎具有语义上的主题,但许多簇都太大而实际上很难是均匀的

    # 为了得到更加均匀的簇,我们可以再次运行算法,这次使用40个簇,并挑选出一些特别有趣的簇
    agglomerative = AgglomerativeClustering(n_clusters=40)
    labels_agg = agglomerative.fit_predict(X_pca)
    print("cluster sizes agglomerative clustering: {}".format(np.bincount(labels_agg)))

    n_clusters = 40
    for cluster in [10, 13, 19, 22, 36]:
        mask = labels_agg == cluster
        cluster_size = np.sum(mask)
        fig, axes = plt.subplots(1, 15, subplot_kw={'xticks': (), 'yticks': ()},figsize=(15, 8))
        axes[0].set_ylabel("#{}: {}".format(cluster, cluster_size))
        for image, label, asdf, ax in zip(X_people[mask], y_people[mask], labels_agg[mask], axes):
            ax.imshow(image.reshape(image_shape), cmap='gray', vmin=0, vmax=1)
            ax.set_title(people.target_names[label].split()[-1],fontdict={'fontsize': 9})
        for i in range(cluster_size, 15):
            axes[i].set_visible(False)
        plt.savefig("./../img/3.无监督学习与预处理/47.将簇的数量设置为40时,从凝聚聚类找到的簇中挑选的图像——左侧文本表示簇的编号和簇中的点的总数_"+str(cluster)+".png", dpi=1080)
        plt.show()



