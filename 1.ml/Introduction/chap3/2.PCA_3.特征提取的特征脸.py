import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # 3.特征提取的特征脸
    """
    PCA特征提取:
        以找到一种数据表示,比给定的原始表示更适合于分析  最经典应用在图像领域
        
    数据集:
        Wild数据集Labeled Faces  使用这些图像的灰度版本,并将它们按比例缩小以加快处理速度
        plt显示照片偏绿色,是因为Matplotlib的默认显示色彩映射(colormap)或图像通道处理方式导致
            1.Matplotlib 的默认色彩映射问题  cmap='viridis' ==> cmap='gray'
            2.数据通道修改为单通道  lfw_people.images[0][:, :, 0]
    """
    people = fetch_lfw_people(data_home ="./../data", min_faces_per_person=20, resize=0.7)
    print(people.data.shape)        # (3023, 5655)
    print(people.target.shape)      # (3023,)
    print(people.images.shape)      # (3023, 87, 65) 3023张图片,每张图片像素87*65,属于62个不同的人  只加载>=20张人脸的人的图像
    print(people.target_names.shape)# (62,)

    image_shape = people.images[0].shape
    fig, axes = plt.subplots(2, 5, figsize=(15, 8), subplot_kw={'xticks': (), 'yticks': ()})
    for target, image, ax in zip(people.target, people.images, axes.ravel()):
        ax.imshow(image,cmap='gray')
        ax.set_title(people.target_names[target])
    plt.savefig("./../img/3.无监督学习与预处理/7.来自Wild数据集中LabeledFaces的一些图像.png", dpi=1080)
    plt.show()

    # 数据有倾斜,每个人最多取50张图片
    counts = np.bincount(people.target) # (62,)
    for i, (count, name) in enumerate(zip(counts, people.target_names)): # zip将两个列表的元素进行配对并添加索引,从0开始
        print("{0:25} {1:3}".format(name, count), end='   ')       # i为索引,(count, name)为zip结果
        if (i + 1) % 3 == 0:                                             # 第1个参数name,左对齐,占据25个字符 第2个参数右对齐,占据3个字符
            print()

    # 使用掩码筛选数据 归一化数据:X_people  数据对应标签:y_people
    mask = np.zeros(people.target.shape, dtype=np.bool)     # 创建掩码 [False ... False] (3023,) 3023个样本
    for target in np.unique(people.target):                 # 每个类别选前50个  np.unique(people.target):0到61
        mask[np.where(people.target == target)[0][:50]] = 1 # 构建掩码 每个类别的前50个样本对应的位置为True,其余为False
    X_people = people.data[mask]                            # 使用掩码筛选数据
    y_people = people.target[mask]
    X_people = X_people / 255.                              # 数据标准化 归一化到[0,1]之间  图像的像素范围为[0,255]


    """
    人脸识别:
        看某个前所未见的人脸是否属于数据库中的某个已知人物
        
    解决方案:
        构建一个分类器,每个人都是一个单独的类别一个人对应一个类别
        
    困难:
        人脸数据库中人数众多,但是同一个人的图像很少,即每个类别的训练样例很少使得模型训练困难
        同时需要不断往数据库中添加人脸,故不需要重新训练一个大型模型
        
    简单的解决方案:
        单一最近邻分类器 
            寻找与你要分类的人脸最为相似的人脸  精度26.6%(每识别四次仅正确识别了一个人)
        PCA
            使用像素距离度量人脸的相似度,使用沿着主成分方向的距离可以提高精度
        PCA白化    
            将主成分缩放到相同的尺度,变换后的结果与使用StandardScaler相同
            旋转并缩放数据,对应缩放形状为圆形
    """
    # PCA白化 mglearn.plots.plot_pca_whitening()
    random_state = np.random.RandomState(5)
    A_ = random_state.normal(size=(300, 2))
    B_ = random_state.normal(size=(2, 2))
    C_ = random_state.normal(size=2)
    X_blob = np.dot(A_,B_) +C_

    pca_white = PCA(whiten=True)
    pca_white.fit(X_blob)
    X_pca_white = pca_white.transform(X_blob)
    print(X_pca_white.shape)

    fig, axes = plt.subplots(1, 2, figsize=(10, 10))
    axes = axes.ravel()
    # 原始数据
    axes[0].set_title("Original data")
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_pca_white[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].set_aspect('equal')     # 设置坐标轴的纵横比 x轴长度=y轴长度
    # PCA白化后的数据
    axes[1].set_title("Whitened data")
    axes[1].scatter(X_pca_white[:, 0], X_pca_white[:, 1], c=X_pca_white[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_xlim(-3, 4)
    plt.savefig("./../img/3.无监督学习与预处理/8.利用启用白化的PCA进行数据变换.png", dpi=1080)
    plt.show()

    # 使用原始数据直接进行分类          KNeighborsClassifier 进行人脸识别
    X_train, X_test, y_train, y_test = train_test_split( X_people, y_people, stratify=y_people, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    print("Test set score of 1-nn: {:.2f}".format(test_score))  # 0.23 1-nn表示使用knn算法,其中k=1  精度不算太差也不不算太好

    # 对数据进行PCA白化处理后进行分类  PCA白化 进行人脸识别
    pca = PCA(n_components=100, whiten=True, random_state=0).fit(X_train)
    X_train_pca = pca.transform(X_train)        # (1547, 5655) ==> (1547, 100) 前100个主成分
    X_test_pca = pca.transform(X_test)          # (516, 5655)  ==> (516, 100)
    knn = KNeighborsClassifier(n_neighbors=1)   # 对新表示使用单一最近邻分类器来
    knn.fit(X_train_pca, y_train)
    test_score = knn.score(X_test_pca, y_test)
    print("Test set accuracy:", np.round(test_score,decimals=2)) # 0.31 精度提高 主成分可能提供了一种更好的数据表示

    # PCA的主成分
    print("pca.components_.shape:",pca.components_.shape)   # (100, 5655)

    fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (component, ax) in enumerate(zip(pca.components_, axes.ravel())):
        ax.imshow(component.reshape(image_shape), cmap='viridis')
        ax.set_title("{}.component".format((i + 1)))
    plt.savefig("./../img/3.无监督学习与预处理/9.人脸数据集前15个主成分的成分向量.png", dpi=1080)
    plt.show()
    """
    虽然我们肯定无法理解这些成分的所有内容,但可以猜测一些主成分捕捉到了人脸图像的哪些方面.
        第一个主成分似乎主要编码的是人脸与背景的对比
        第二个主成分编码的是人脸左半部分和右半部分的明暗程度差异,如此等等.
    虽然这种表示比原始像素值的语义稍强,但它仍与人们感知人脸的方式相去甚远.
    由于PCA模型是基于像素的,因此人脸的相对位置(眼睛,下巴和鼻子的位置)和明暗程度都对两张图像在像素表示中的相似程度有很大影响.
    但人脸的相对位置和明暗程度可能并不是人们首先感知的内容.
    在要求人们评价人脸的相似度时,他们更可能会使用年龄,性别,面部表情和发型等属性,而这些属性很难从像素强度中推断出来.
    重要的是要记住,算法对数据(特别是视觉数据,比如人们非常熟悉的图像)的解释通常与人类的解释方式大不相同
    """
    # 总结: 算法对数据的解释通常与人类的解释方式大不相同,特别是视觉数据

    """
    PCA变换:
        先旋转数据,然后删除方差较小的成分   见书本截图
        
    PCA模型:
        理解方式一:
            找到一些数字(PCA旋转后的新特征值),将测试点表示为主成分的加权求和  ===>  将图像分解为成分的加权求和
                target = x[0]*components_[0] + x[1]*components_[1] + ...
                    x[0],x[1]...为数据点的主成分的系数,他们是图像在旋转后的空间中的表示  

        理解方式二:
            将数据降维到只包含一些主成分,然后反向旋转回到原始空间,即仅使用一些成分对原始数据进行重建
    """
    # mglearn.plots.plot_pca_faces(X_train, X_test, image_shape)
    reduced_images = []
    for n_components in [10, 50, 100, 500]:
        pca = PCA(n_components=n_components).fit(X_train)
        X_test_pca = pca.transform(X_test)
        X_test_back = pca.inverse_transform(X_test_pca)
        reduced_images.append(X_test_back)
    fix, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks':(), 'yticks':()})
    for i, ax in enumerate(axes):
        ax[0].imshow(X_test[i].reshape(image_shape), vmin=0, vmax=1, cmap='gray')
        for a, X_test_back in zip(ax[1:], reduced_images):
            a.imshow(X_test_back[i].reshape(image_shape), vmin=0, vmax=1, cmap='gray')
    axes[0, 0].set_title("original image")
    for ax, n_components in zip(axes[0, 1:], [10, 50, 100, 500]):
        ax.set_title("%d components" % n_components)
    plt.savefig("./../img/3.无监督学习与预处理/11.利用越来越多的主成分对三张人脸图像进行重建.png", dpi=1080)
    plt.show()
    """
    总结:
        1.仅使用前10个主成分时,仅捕捉到图片的基本特点 
        2.随着使用的主成分越来越多,图像中也保留越来越多的细节 ==> 对应求和中包含越来越多的项
        3.如果使用的成分个数与像素个数相等,意味着我们在旋转后不会丢弃任何信息,可以完美重建图像
    """

    # 使用PCA的前两个主成分,将数据集中的所有人脸在散点图中可视化
    mglearn.discrete_scatter(X_train_pca[:, 0], X_train_pca[:, 1], y_train)
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.savefig("./../img/3.无监督学习与预处理/12.利用前两个主成分绘制人脸数据集的散点图.png", dpi=1080)
    plt.show()