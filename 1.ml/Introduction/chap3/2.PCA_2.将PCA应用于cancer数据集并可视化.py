
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    # 2.将PCA应用于cancer数据集并可视化
    """
    特征可视化:
        鸢尾花 3个特征  散点图矩阵
        癌症集 30个特征 每个特征分别计算两个类别的直方图  ===>  PCA最常见的应用之一就是将高维数据集可视化

    总结:
        每个特征创建一个直方图,计算具有某一特征的数据点在特定范围内(bin)的出现频率
            了解每个特征在两个类别中的分布情况,猜测哪些特征能够更好地区分良性样本和恶性样本
            eg:
                "smoothness error"特征似乎没有什么信息量,因为两个直方图大部分都重叠在一起
                "worst concave points"特征看起来信息量相当大,因为两个直方图的交集很小
                
        但无法展示变量之间的相互作用以及这种相互作用与类别之间的关系
        利用PCA,我们可以获取到主要的相互作用,并得到稍为完整的图像,找到前两个主成分,并在这个新的二维空间中用散点图将数据可视化
    """
    cancer = load_breast_cancer()
    malignant = cancer.data[cancer.target == 0] # 恶行
    benign = cancer.data[cancer.target == 1]    # 良性
    fig, axes = plt.subplots(15, 2, figsize=(10, 20))
    ax = axes.ravel()
    for i in range(30):
        _, bins = np.histogram(cancer.data[:, i], bins=50)  # 30个特征对,对应的特征范围划分50份
        ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)  # 恶性第i个特征的直方图
        ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)     # 良性第i个特征的直方图
        ax[i].set_title(cancer.feature_names[i])
        ax[i].set_yticks(())
    ax[0].set_xlabel("Feature magnitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant", "benign"], loc="best")
    fig.tight_layout()
    plt.savefig("./../img/3.无监督学习与预处理/4.乳腺癌数据集中每个类别的特征直方图.png",dpi=1080)
    plt.show()

    """
    PCA降维:
        对PCA对象实例化,fit()找到主成分,transform()旋转并降维
        默认情况下,PCA仅旋转(并移动)数据,保留所有主成分,为降低数据的维度,创建PCA对象时需指定保留的主成分个数
    """
    # 利用PCA,获取到主要的相互作用,找到前两个主成分,并在这个新的二维空间中用散点图将数据可视化
    X_scaled = StandardScaler().fit(cancer.data).transform(cancer.data) # (569, 30)
    pca_n2 = PCA(n_components=2).fit(X_scaled)
    X_pca_n2 = pca_n2.transform(X_scaled)                               # (569, 2)

    plt.figure(figsize=(8, 8))
    mglearn.discrete_scatter(X_pca_n2[:, 0], X_pca_n2[:, 1], cancer.target)
    plt.legend(cancer.target_names, loc="best")
    plt.gca().set_aspect("equal")
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.savefig("./../img/3.无监督学习与预处理/5.利用前两个主成分绘制乳腺癌数据集的二维散点图.png", dpi=1080)
    plt.show()
    """
    总结:
        PCA是一种无监督方法,在寻找旋转方向时没有用到任何类别信息.它只是观察数据中的相关性
        此图中绘制了第一主成分与第二主成分的关系,然后利用类别信息对数据点进行着色
            在这个二维空间中两个类别被很好的分离,这让我们相信,即使是线性分类器(在这个空间中学习一条直线)也可以在区分这个两个类别时表现得相当不错
            我们还可以看到,恶性点比良性点更加分散,直方图可看出
    """

    """
    PCA缺点:
        不容易对图中的两个轴做出解释,主成分对应于原始数据中的方向,所以它们是原始特征的组合,但这些组合往往非常复杂
        在拟合过程中,主成分被保存在PCA.components_属性中       
    """
    print("PCA component shape:",pca_n2.components_.shape)  # (2, 30) 每行对应一个主成分,按重要性排序,列对应PCA的30个原始特征
    print("PCA components:",np.round(pca_n2.components_,decimals=3))

    # 用热图将系数可视化 乳腺癌数据集前两个主成分的热图
    plt.figure(figsize=(12, 6))
    plt.matshow(pca_n2.components_, cmap='viridis', fignum=0)
    plt.xticks(range(len(cancer.feature_names)), cancer.feature_names, rotation=90, ha='left')
    plt.yticks([0, 1], ["First component", "Second component"])
    plt.colorbar()
    plt.xlabel("Feature")
    plt.ylabel("Principal components")
    plt.savefig("./../img/3.无监督学习与预处理/6.乳腺癌数据集前两个主成分的热图.png", dpi=1080)
    plt.show()
    """
    总结:
        第一个主成分所有特征的符号相同(箭头指向哪个方向无关紧要),所有特征之间存在普遍的相关性
        第二个主成分的符号有正有负,两个主成分都包含所有30个特征,这种所有特征的混合使得解释热力图的坐标轴变得十分困难
    """


