
import mglearn
import numpy as np
from cffi.cffi_opcode import PRIM_SHORT
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

"""
    无监督学习进行数据变换,方便可视化与压缩数据,以便寻找信息量更大的数据表示用于进一步的处理
        降维          特征提取        流形学习
        PCA           NMF           t-SNE 
        最常用方法      特征提取       二维散点可视化    
"""
if __name__ == '__main__':
    # 1.主成分分析 PCA
    """
    PCA:
        一种旋转数据集的方法,旋转后的特征在统计上不相关
        数据集旋转后,根据新特征对解释数据的重要性来选择它的一个子集
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()                     # 数组(2, 2)展开为一维数组 (4,)

    random_state = np.random.RandomState(5)  # 设置随机种子
    A_ = random_state.normal(size=(300, 2))  # 标准正态分布 (300,2)与(2,2)
    B_ = random_state.normal(size=(2, 2))
    C_ = random_state.normal(size=2)
    X_blob = np.dot(A_,B_) + C_              # 对数据进行线性变换(相当于旋转和缩放)和位移(添加随机偏移量)
    # 生成300个数据点,大致呈现高斯分布(blob),因为线性变换所以blob可能是椭圆形 这类数据用于测试聚类算法

    pca = PCA().fit(X_blob)
    X_pca = pca.transform(X_blob)           # (300, 2)
    std = X_pca.std(axis=0)                 # [2.53854851 0.38087708]

    # 原始数据 X_blob
    axes[0].scatter(X_blob[:, 0], X_blob[:, 1], c=X_pca[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[0].set_title("Original data")
    axes[0].set_xlabel("feature 1")
    axes[0].set_ylabel("feature 2")
    axes[0].text(-1.5, -.5, "Component 2", size=14)
    axes[0].text(-4, -4, "Component 1", size=14)
    axes[0].set_aspect('equal')
    axes[0].arrow(pca.mean_[0], pca.mean_[1],           # 起始位置(数据均值点)
                  pca.components_[0, 0] * std[0] ,       # x方向增量(第1主成分的x方向权重*标准差)
                  pca.components_[0, 1] * std[0] ,       # y方向增量(第2主成分的x方向权重*标准差)
                  width=.1, head_width=.3, color='k')
    axes[0].arrow(pca.mean_[0], pca.mean_[1],
                  pca.components_[1, 0] * std[1] ,
                  pca.components_[1, 1] * std[1] ,
                  width=.1, head_width=.3, color='k')
    # 在同一个坐标轴axes[0]上绘制两箭头,用于可视化PCA的两个主成分方向及其方差大小
    """
    在同一个图上绘制了PCA的两个主成分方向 
        PC1:数据变化最大的方向(方差最大)
        PC2:与PC1正交(垂直),数据变化第二大的方向(方差次大)
    """

    # 使用PCA进行降维/旋转 X_pca
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                    c=X_pca[:, 0],      # 点的颜色随第一主成分的值变化
                    linewidths=0, s=60,
                    cmap='viridis')
    axes[1].set_title("Transformed data")
    axes[1].set_xlabel("First principal component")
    axes[1].set_ylabel("Second principal component")
    axes[1].set_aspect('equal')
    axes[1].set_ylim(-8, 8)

    # 使用PCA进行降维后,仅保留第一主成分
    axes[2].set_title("Transformed data w/second component dropped")
    axes[2].scatter(X_pca[:, 0],
                    np.zeros(X_pca.shape[0]),   # 创建全零数组 np.zeros y=0
                    c=X_pca[:, 0],
                    linewidths=0, s=60,
                    cmap='viridis')
    axes[2].set_xlabel("First principal component")
    axes[2].set_aspect('equal')
    axes[2].set_ylim(-8, 8)

    # 逆旋转并将平均值添加到数据中
    pca_n1 = PCA(n_components=1).fit(X_blob)
    X_pca_n1 = pca_n1.transform(X_blob)
    X_inverse = pca_n1.inverse_transform(X_pca_n1)
    axes[3].set_title("Back-rotation using only first component")
    axes[3].scatter(X_inverse[:, 0], X_inverse[:, 1], c=X_pca[:, 0], linewidths=0, s=60, cmap='viridis')
    axes[3].set_xlabel("feature 1")
    axes[3].set_ylabel("feature 2")
    axes[3].set_aspect('equal')
    axes[3].set_xlim(-8, 4)
    axes[3].set_ylim(-8, 4)

    plt.savefig("./../img/3.无监督学习与预处理/3.用PCA做数据变换.png",dpi=1080)
    plt.show()

    """
    总结:
        第一张:
            原始数据点,使用不同颜色区分 
                首先找到方差最大的方向标记为成分1,为数据中包含信息最多的方向(向量)
                    ===>沿着这个方向的特征之间最为相关
                其次找到与第一个方向正交(成直角)且包含最多信息的方向
                    ===>在二维空间中,只有一个成直角的方向,但在更高维的空间中会有(无穷)多的正交方向
            此过程中找到的方向为主成分,为数据方差的主要方向,一般主成分的个数与原始特征相同
        
        第二张:
            使用PCA对数据进行处理,首先将数据减去均值,随后旋转数据,使得第一主成分//x轴,第二主成分//y轴
                ===>使得变换后的数据以零为中心

        第三张:
            使用PCA降维,仅保留第一主成分,数据从二维降为一维
            
        第四张:   
            在原始空间中获得去掉第二个成分的新数据点,仅包含第一主成分
            反向旋转并将平均值重新加到数据中,这些数据点位于原始特征空间中,但仅保留了第一主成分中包含的信息     
            这种变换有时用于去除数据中的噪声影响,或者将主成分中保留的那部分信息可视化
    """