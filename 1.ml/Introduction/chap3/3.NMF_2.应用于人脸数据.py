import mglearn
import numpy as np
from matplotlib import pyplot as plt
from mglearn.plot_nmf import nmf_faces
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import NMF
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    # 2.将NMF应用于人脸数据 NMF主要参数为提取的分量个数,需小于输入特征的个数
    # 1.观察分量个数如何影响NMF重建数据的好坏
    people = fetch_lfw_people(data_home ="./../data", min_faces_per_person=20, resize=0.7) # fetch的数据就是偏绿色
    image_shape = people.images[0].shape
    mask = np.zeros(people.target.shape, dtype=np.bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1
    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255.
    X_train, X_test, y_train, y_test = train_test_split( X_people, y_people, stratify=y_people, random_state=0)

    # mglearn.plots.plot_nmf_faces(X_train, X_test, image_shape)
    reduced_images = []
    for n_components in [10, 50, 100, 500]:
        nmf = NMF(n_components=n_components, random_state=0)
        nmf.fit(X_train)
        X_test_nmf = nmf.transform(X_test)
        X_test_back = np.dot(X_test_nmf, nmf.components_)
        reduced_images.append(X_test_back)

    # 原始图像 vs 使用NMF提取的特征重建图像
    fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
    for i, ax in enumerate(axes):
        ax[0].imshow(X_test[i].reshape(image_shape), vmin=0, vmax=1, cmap='gray')
        for a, X_test_back in zip(ax[1:], reduced_images):
            a.imshow(X_test_back[i].reshape(image_shape), vmin=0, vmax=1, cmap='gray')
    axes[0, 0].set_title("original image")
    for ax, n_components in zip(axes[0, 1:], [10, 50, 100, 500]):
        ax.set_title("%d components" % n_components)
    plt.savefig("./../img/3.无监督学习与预处理/14.利用越来越多分量的NMF重建三张人脸图像.png", dpi=1080)
    plt.show()

    """
    总结:
        反向变换的数据质量与使用PCA时类似,但要稍差一些,因为PCA找到的是重建的最佳方向
        NMF通常并不用于对数据进行重建或编码,而是用于在数据中寻找有趣的模式
    """

    # 2.提取1到15个分量观察数据
    nmf = NMF(n_components=15, random_state=0)
    nmf.fit(X_train)
    X_train_nmf = nmf.transform(X_train)
    X_test_nmf = nmf.transform(X_test)
    fig, axes = plt.subplots(3, 5, figsize=(15, 12),subplot_kw={'xticks': (), 'yticks': ()})
    for i, (component, ax) in enumerate(zip(nmf.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='gray')
        ax.set_title("{}. component".format(i))
    plt.savefig("./../img/3.无监督学习与预处理/15.使用15个分量的NMF在人脸数据集上找到的分量.png", dpi=1080)
    plt.show()

    # 书本说分量3与7可以看出向左/向右转动的人脸,故查看分量3与7
    compn = 3
    inds = np.argsort(X_train_nmf[:, compn])[::-1] # 数组值降序排列对应的索引 [ 947 1245  737 ...  619 1415  920] (1547,)
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    fig.suptitle("Large component 3")
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape), cmap='gray')
    plt.savefig("./../img/3.无监督学习与预处理/16.分量3系数较大的人脸.png", dpi=1080)
    plt.show()

    compn = 7
    inds = np.argsort(X_train_nmf[:, compn])[::-1]
    fig.suptitle("Large component 7")
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (ind, ax) in enumerate(zip(inds, axes.ravel())):
        ax.imshow(X_train[ind].reshape(image_shape), cmap='gray')
    plt.savefig("./../img/3.无监督学习与预处理/17.分量7系数较大的人脸.png", dpi=1080)
    plt.show()

    """
    总结:
        分量3系数较大的人脸都是向右看的人脸,而分量7系数较大的人脸都向左看
        提取这样的模式最适合于具有叠加结构的数据,包括音频,基因表达和文本数据
    """