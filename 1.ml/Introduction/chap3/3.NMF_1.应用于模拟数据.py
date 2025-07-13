
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF

if __name__ == '__main__':
    """
    NMF:
        非负矩阵分解,用于提取有用的特征或降维
        同PCA将每个数据点写成一些分量的加权求和
        
        特点:
            减少分量个数不仅会删除一些方向,而会创建一组完全不同的分量
            NMF的分量不排序,所有分量的地位平等 
        
        适用场景:
            将数据分解成非负加权求和的这个过程,对由多个独立源相加(或叠加)创建而成的数据特别有用
        
        NMF vs PCA:
            PCA:需要的是正交分量,并且能够解释尽可能多的数据方差
            NMF:分量和系数均为非负,仅用于每个特征都是非负的数据   
                减少分量个数不仅会删除一些方向,而且会创建一组完全不同的分量 
                分量没有排序,地位均等
    """

    # 1.NMF用于模拟数据,需保证数据是正的
    random_state = np.random.RandomState(5)
    A_ = random_state.normal(size=(300, 2))
    B_ = random_state.normal(size=(2, 2))
    C_ = random_state.normal(size=2)
    X_blob = np.dot(A_,B_) + C_ + 8  # 保证数据都是正的

    nmf = NMF(random_state=0).fit(X_blob)
    X_nmf = nmf.transform(X_blob)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_nmf[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].set_xlim(0, 12)
    axes[0].set_ylim(0, 12)
    axes[0].arrow(0, 0, nmf.components_[0, 0], nmf.components_[0, 1], width=.1,  head_width=.3, color='k')
    axes[0].arrow(0, 0, nmf.components_[1, 0], nmf.components_[1, 1], width=.1,  head_width=.3, color='k')
    axes[0].set_aspect('equal')
    axes[0].set_title("NMF with two components")

    nmf_n1 = NMF(random_state=0, n_components=1)
    nmf_n1.fit(X_blob)
    axes[1].scatter(X_blob[:, 0], X_blob[:, 1], c=X_nmf[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[1].set_xlabel("feature 1")
    axes[1].set_ylabel("feature 2")
    axes[1].set_xlim(0, 12)
    axes[1].set_ylim(0, 12)
    axes[1].arrow(0, 0, nmf_n1.components_[0, 0], nmf_n1.components_[0, 1], width=.1,  head_width=.3, color='k')
    axes[1].set_aspect('equal')
    axes[1].set_title("NMF with one component")
    plt.savefig("./../img/3.无监督学习与预处理/13.两个分量的非负矩阵分解(左)和一个分量的非负矩阵分解(右)找到的分量.png", dpi=1080)
    plt.show()

    """
    NMF数据:
        需要保证数据是正的,说明数据相对于原点(0,0)的位置实际上对NMF很重要
        可将提取出来的非负分量看作是从(0,0)到数据的方向
        
    两个分量的NMF:
        所有数据点都可写成这两个分量的正数组合 
        若有足够多的分量则能够完美重建数据(分量个数与特征个数相同),算法会选择指向数据极值的方向
        
    一个分量的NMF:
        NMF创建一个指向平均值的分量,方便对数据做出最合理的解释
    """
