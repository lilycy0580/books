from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

if __name__ == '__main__':
    """
    PCA:
        通常是用于变换数据的首选方法,使你能够用散点图将其可视化,但是先旋转然后减少方向限制其有限性
    
    流形学习算法:
        允许进行更复杂的映射,可以给出更好的可视化  t-SNE
        
    t-SNE:
        思想:
            找到数据的一个二维表示,尽可能地保持数据点之间的距离
            给出每个数据点的随机二维表示,然后尝试让在原始特征空间中距离较近的点更加靠近,原始特征空间中相距较远的点更加远离
        
        重点关注:
            距离较近的点,而不是保持距离较远的点之间的距离,保存那些表示哪些点比较靠近的信息
            
        参数:
            默认参数的效果通常就很好
            可尝试修改perplexity 和 early_exaggeration,但效果较小
    """
    # 原始数据
    digits = load_digits()  #  digits.data:(1797, 64)  digits.target:(1797,)  digits.images:(1797, 8, 8)
    fig, axes = plt.subplots(2, 5, figsize=(10, 5), subplot_kw={'xticks':(), 'yticks': ()})
    for ax, img in zip(axes.ravel(), digits.images):
        ax.imshow(img,cmap='gray')
    plt.savefig("./../img/3.无监督学习与预处理/20.digits数据集的示例图像.png", dpi=1080)
    plt.show()

    # PCA降维数据可视化 
    digits_pca = PCA(n_components=2).fit(digits.data).transform(digits.data)
    colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
    plt.figure(figsize=(10, 10))
    plt.xlim(digits_pca[:, 0].min(), digits_pca[:, 0].max())
    plt.ylim(digits_pca[:, 1].min(), digits_pca[:, 1].max())
    for i in range(len(digits.data)):
        # 在二维图形上添加文本标签
        plt.text(digits_pca[i, 0],                  # x坐标:第i个样本在PCA降维后的第一个主成分值
                 digits_pca[i, 1],                  # y坐标:第i个样本在PCA降维后的第二个主成分值
                 str(digits.target[i]),             # 显示的文本:第i个样本的真实标签（转换为字符串）
                 color=colors[digits.target[i]],    # 颜色:根据标签从colors数组中选取对应颜色
                 fontdict={'weight': 'bold', 'size': 9}) # 字体样式:略
    plt.xlabel("First principal component")
    plt.ylabel("Second principal component")
    plt.savefig("./../img/3.无监督学习与预处理/21.利用前两个主成分绘制digits数据集的散点图.png", dpi=1080)
    plt.show()
    """
    总结:
        用每个类别对应的数字作为符号来显示每个类别的位置
        利用前两个主成分可以将数字0,6和4相对较好地分开,尽管仍有重叠.大部分其他数字都大量重叠在一起
    """

    # t-SNE降维(不支持变换新数据)
    # use fit_transform instead of fit, as TSNE has no transform method
    digits_tsne = TSNE(random_state=42).fit_transform(digits.data)
    plt.figure(figsize=(10, 10))
    plt.xlim(digits_tsne[:, 0].min(), digits_tsne[:, 0].max() + 1)
    plt.ylim(digits_tsne[:, 1].min(), digits_tsne[:, 1].max() + 1)
    for i in range(len(digits.data)):
        # 在二维图形上添加文本标签
        plt.text(digits_tsne[i, 0], digits_tsne[i, 1], str(digits.target[i]),
                 color = colors[digits.target[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.xlabel("t-SNE feature 0")
    plt.ylabel("t-SNE feature 1")
    plt.savefig("./../img/3.无监督学习与预处理/22.利用t-SNE找到的两个分量绘制digits数据集的散点图.png", dpi=1080)
    plt.show()
    """
    总结:
        所有类别都被明确分开。数字 1 和 9 被分成几块,但大多数类别都形成一个密集的组
        这种方法并不知道类别标签:它完全是无监督的,但它能够找到数据的一种二维表示,仅根据原始空间中数据点之间的靠近程度就能够将各个类别明确分开
    """





