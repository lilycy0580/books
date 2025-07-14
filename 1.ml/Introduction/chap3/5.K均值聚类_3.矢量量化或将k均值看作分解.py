import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_lfw_people, make_moons
from sklearn.decomposition import NMF, PCA
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """
    数据点表示为一些分量之和:
        PCA 试图找到数据中方差最大的方向 对应于数据的“极值”
        NMF 试图找到累加的分量         对应于数据的“部分”

    用簇中心来表示每个数据点:
        k均值 将其看作仅用一个分量来表示每个数据点,该分量由簇中心给出
              将k均值看作是一种分解方法,其中每个点用单一分量来表示       矢量量化

    """
    # 案例一:比较PCA,NMF和K均值各自提取的分量 及 利用100个分量对测试集中人脸的重建
    people = fetch_lfw_people(data_home ="./../data", min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255.
    X_train, X_test, y_train, y_test = train_test_split(X_people, y_people, stratify=y_people, random_state=0)

    pca = PCA(n_components=100, random_state=0).fit(X_train)
    pca_X_test = pca.transform(X_test)
    rebuild_X_pca = pca.inverse_transform(pca_X_test)

    nmf = NMF(n_components=100, random_state=0).fit(X_train)
    nmf_X_test = nmf.transform(X_test)
    rebuild_X_nmf = np.dot(nmf_X_test, nmf.components_)

    km = KMeans(n_clusters=100, random_state=0).fit(X_train)
    km_X_test = km.predict(X_test)
    rebuild_X_km = km.cluster_centers_[km_X_test]

    fig, axes = plt.subplots(3, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
    fig.suptitle("Extracted Components")
    for ax, pca_comp, nmf_comp, km_comp in zip(axes.T, pca.components_, nmf.components_, km.cluster_centers_):
        ax[0].imshow(pca_comp.reshape(image_shape))
        ax[1].imshow(nmf_comp.reshape(image_shape),cmap='gray')
        ax[2].imshow(km_comp.reshape(image_shape),cmap='gray')
    axes[0, 0].set_ylabel("pca")
    axes[1, 0].set_ylabel("nmf")
    axes[2, 0].set_ylabel("kmeans")
    plt.savefig("./../img/3.无监督学习与预处理/30.对比k均值的簇中心与PCA和NMF找到的分量.png", dpi=1080)
    plt.show()

    fig, axes = plt.subplots(4, 5, subplot_kw={'xticks': (), 'yticks': ()},figsize=(8, 8))
    fig.suptitle("Reconstructions")
    for ax, orig, rebuild_pca, rebuild_nmf, rebuild_km in zip(axes.T, X_test, rebuild_X_pca, rebuild_X_nmf, rebuild_X_km):
        ax[0].imshow(orig.reshape(image_shape),cmap='gray')
        ax[2].imshow(rebuild_pca.reshape(image_shape),cmap='gray')
        ax[3].imshow(rebuild_nmf.reshape(image_shape),cmap='gray')
        ax[1].imshow(rebuild_km.reshape(image_shape),cmap='gray')
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("pca")
    axes[2, 0].set_ylabel("nmf")
    axes[3, 0].set_ylabel("kmeans")
    plt.savefig("./../img/3.无监督学习与预处理/31.利用100个分量(或簇中心)的k均值,PCA和NMF的图像重建的对比——k均值的每张图像中仅使用了一个簇中心.png", dpi=1080)
    plt.show()


    # 案例二:k均值做矢量量化  用比输入维度更多的簇来对数据进行编码 若使用PCA或NMF降维则会完全破坏数据的结构
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)    # (200, 2) (200,)
    km = KMeans(n_clusters=10, random_state=0).fit(X)               # 10个簇中心,每个点被分配0到9之间的一个数字
    y_pred = km.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')    # 使用'Paired'颜色映射
    plt.scatter(km.cluster_centers_[:, 0], km.cluster_centers_[:, 1], s=60, marker='H', c='black', linewidth=2)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/32.利用k均值的许多簇来表示复杂数据集中的变化.png", dpi=1080)
    plt.show()
    print("Cluster memberships:",y_pred)

    distance_features = km.transform(X)
    print("Distance feature shape: {}".format(distance_features.shape)) # (200, 10)
    print("Distance features:\n{}".format(distance_features))
    print("Cluster centers:\n{}".format(km.cluster_centers_))
    """
    总结:
        10个簇,将每个数据点看作为10个分量表示的数据(10个新特征),只有表示该点对应的簇中心的那个特征不为0,其他特征均为0
        利用这个10维表示,现在可以用线性模型来划分两个半月形,而利用原始的两个特征是不可能做到这一点的.
        将到每个簇中心的距离作为特征,还可以得到一种表现力更强的数据表示  transform
    """


