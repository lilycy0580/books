
# 核支持向量机  处理分类/回归任务 Kernelized Support Vector Machines
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs, load_breast_cancer
import mglearn
from mpl_toolkits.mplot3d import Axes3D, axes3d
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

if __name__ == '__main__':
    # 1.线性模型与非线性特征
    # 线性模型在低维空间中可能非常受限,因为线和平面的灵活性有限 需要增加输入特征的交互项或多项式
    X, y = make_blobs(centers=4, random_state=8)
    y = y % 2
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/2.监督学习/36.二分类数据集,其类别并不是线性可分的.png",dpi=1080)
    plt.show()
    # 总结:用于分类的线性模型只能用一条直线来划分数据点,对这个数据集无法给出较好的结果

    linear_svm = LinearSVC().fit(X, y)
    mglearn.plots.plot_2d_separator(linear_svm, X)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/2.监督学习/37.线性SVM给出的决策边界.png",dpi=1080)
    plt.show()

    # 对输入特征进行扩展 添加交互项或多项式  二维数据变三维
    X_new = np.hstack([X, X[:, 1:] ** 2])       # (100, 3)   [[a,b,c]]--->[[a.b,c,b^2,c^2]]
    mask = y == 0                               # (100,)   [False True False.....True] 创建一个布尔掩码
    figure = plt.figure()                       # 构建画图对象figure
    # 给 figure对象添加一个子图
    ax = figure.add_subplot(111,                # 1行1列网格中的第1个子图
                            projection='3d',    # 指定这是一个3D子图,创建一个三维坐标系,允许绘制3D图形
                            elev=-152,          # 设置仰角(elevation)为-152度
                            azim=-26)           # 设置方位角(azimuth)为-26度
    # 在3D坐标系中绘制散点图
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2],
               c='b',
               cmap=mglearn.cm2,
               s=60,
               edgecolor='k')
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2],
               c='r',                           # 设置散点的颜色为红色
               marker='^',                      # 设置散点的标记形状为三角形
               cmap=mglearn.cm2,                # 若c为数值数组,则使用此颜色映射
               s=60,                            # 设置散点的大小为60
               edgecolor='k')                   # 设置散点边缘颜色为黑色
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1**2")
    plt.savefig("./../img/2.监督学习/38.对图2-37中的数据集进行扩展,新增由feature1导出的的第三个特征.png",dpi=1080)
    plt.show()
    # 总结:添加多项式/交互项后,可用线性模型(三维空间中的平面)将这两个类别分开

    # 线性SVM绘制决策边界
    linear_svm_3d = LinearSVC().fit(X_new, y)
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_     # 斜率,轴距
    figure = plt.figure()
    ax = Axes3D(figure, elev=-152, azim=-26)
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
    XX, YY = np.meshgrid(xx, yy)    # 构建二维网格坐标矩阵
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]                   # a*X[0] + b*X[1] + c*X[2] + d = 0
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',cmap=mglearn.cm2, s=60, edgecolor='k')
    ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',cmap=mglearn.cm2, s=60, edgecolor='k')
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
    ax.set_zlabel("feature1 ** 2")
    plt.savefig("./../img/2.监督学习/39.线性SVM对扩展后的三维数据集给出的决策边界.png",dpi=1080)
    plt.show()

    # 决策边界为椭圆
    ZZ = YY ** 2    # 构建抛物面 z = y^2,将三维决策边界投影到二维平面上
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()], cmap=mglearn.cm2, alpha=0.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/2.监督学习/40.将图2-39给出的决策边界作为两个原始特征的函数.png",dpi=1080)
    plt.show()

    # 2.核技巧
    """
    核技巧:kernel trick
        向数据表示中添加非线性特征,可以让线性模型变得更强大,但伴随着不知道要添加哪些特征以及添加后计算开销大
        核技巧是直接计算扩展特征表示中数据点之间的距离(内积),以便在高维空间中学习分类器,而不用实际计算可能非常大的新的数据表示

    支持向量机映射到高维空间:
        1.多项式核 在一定阶数内计算原始特征所有可能的多项式 fea1**5,fea2**5
        2.RBF核 高斯核,考虑所有阶数的所有可能的多项式,但阶数越高,特征的重要性越小
    """

    # 3.理解SVM
    """
    理解SVM:
        在训练过程中,SVM学习每个训练数据点对于表示两个类别之间的决策边界的重要性.
        通常只有一部分训练数据点对于定义决策边界来说很重要:位于类别之间边界上的那些点,这些点叫作支持向量

    预测新样本:
        测量新样本点与每个支持向量之间的距离,分类决策是基于它与支持向量之间的距离以及在训练过程中学到的支持向量重要性
        高斯核计算数据点之间的距离
    """
    # 构建数据集,人为制造噪声并随机丢弃部分数据
    X, y = make_blobs(centers=2, random_state=4, n_samples=30) # (30, 2) (30,) 2个簇 [0 0 1 0 1 0 0 1 1 1 0 1 1 1 1 0 0 1 1 1 0 0 1 0 0 0 0 1 1 0]
    y[np.array([7, 27])] = 0            # 人为制造一些噪声或错误标签 y[7],y[27] 1==>0
    mask = np.ones(len(X), dtype=bool)  # 创建掩码,用于后续筛选数据 (30,) [True,...True]
    mask[np.array([0, 1, 5, 26])] = 0   # 后续会丢弃这些样本 True==>False
    X, y = X[mask], y[mask]             # 应用掩码筛选数据 (26, 2) (26,)

    svm = SVC(kernel='rbf',             # 使用径向基函数作为核函数

              C=10,                     # 正则化参数为10 控制模型对误分类的惩罚程度
                                        # C较大表示严格分类,可能过拟合 C较小表示更多误分类,可能欠拟合

              gamma=0.1)                # RBF核函数的系数为0.1
                                        # gamma较大使决策边界更紧贴数据,可能过拟合 gamma较小使决策边界更平滑,可能欠拟合
    svm.fit(X, y)

    sv = svm.support_vectors_           # 获取模型中所有支持向量的坐标,每一行是一个支持向量(决定决策边界的关键数据点)
                                            # svm.dual_coef_:SVM的对偶问题解(即支持向量的系数),正负号表示该支持向量属于哪一类
    sv_labels = svm.dual_coef_.ravel() > 0  # ravel():将多维数组展平为一维 >0 True/False 正负类
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)   # 绘制支持向量,根据lables进行区分
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.savefig("./../img/2.监督学习/41.RBF核SVM给出的决策边界和支持向量.png", dpi=1080)
    plt.show()
    # 总结:SVM给出非常平滑且非线性(非直线)的边界 此处调节gamma与C参数

    # 4.SVM调参
    """
    gamma:
        控制高斯核的宽度,决定了点与点之间"靠近"是指多大的距离
    C:
        正则化参数,与线性模型中用到的类似,用于限制每个点的重要性 dual_coef_
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    for ax, C in zip(axes, [-1, 0, 3]):
        for a, gamma in zip(ax, range(-1, 2)):
            mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
    axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],ncol=4, loc=(.9, 1.2))
    plt.savefig("./../img/2.监督学习/42.设置不同的C和gamma参数对应的决策边界和支持向量.png", dpi=1080)
    plt.show()
    """
    总结:
        gamma:
            gamma值较小,说明高斯核的半径较大,许多点都被看作比较靠近 左侧的图决策边界非常平滑,越向右的图决策边界更关注单个点
            gamma值较小表示决策边界变化很慢,生成的是复杂度较低的模型,gamma值较大则会生成更为复杂的模型
        C:
          C值很小,说明模型非常受限,每个数据点的影响范围都有限 决策边界看起来几乎是线性的,误分类的点对边界几乎没有任何影响
          C值较大,这些点对模型的影响变大,使得决策边界发生弯曲来将这些点正确分类
    """
    
    # RBF核 SVM + cancer C=1,gamma=1/n_features
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split( cancer.data, cancer.target, random_state=0)
    svc = SVC(C=1,gamma=1).fit(X_train, y_train) # 以前默认值,现在默认值C=1,gamma=scale
    train_score = svc.score(X_train, y_train)
    test_score = svc.score(X_test, y_test)
    print("Accuracy on training set: {:.2f}".format(train_score))   # 0.90
    print("Accuracy on test set: {:.2f}".format(test_score))        # 0.63 过拟合
    plt.plot(X_train.min(axis=0), 'o', label="min")
    plt.plot(X_train.max(axis=0), '^', label="max")
    plt.legend(loc=4)
    plt.xlabel("Feature index")
    plt.ylabel("Feature magnitude")
    plt.yscale("log")
    plt.savefig("./../img/2.监督学习/43.乳腺癌数据集的特征范围(注意y轴的对数坐标).png", dpi=1080)
    plt.show()
    # 总结:SVM对参数的设定和数据的缩放非常敏感,要求所有特征有相似的变化范围

    # 5.为SVM预处理数据 对每个特征进行缩放,使其大致都位于同一范围,将所有特征缩放到0和1之间  MinMaxScaler
    train_min = X_train.min(axis=0)                         # min
    train_range = (X_train - train_min).max(axis=0)         # max
    X_train_scaled = (X_train - train_min) / train_range    # range
    print("Minimum for each feature\n", X_train_scaled.min(axis=0)) # [0. 0...]
    print("Maximum for each feature\n", X_train_scaled.max(axis=0)) # [1. 1...]

    X_test_scaled = (X_test - train_min) / train_range
    svc = SVC(C=1,gamma=1).fit(X_train_scaled, y_train)
    train_scaled_score = svc.score(X_train_scaled, y_train)
    test_scaled_score = svc.score(X_test_scaled, y_test)
    print("Accuracy on training set: {:.3f}".format(train_scaled_score))    # 0.984
    print("Accuracy on test set: {:.3f}".format(test_scaled_score))         # 0.972

    svc = SVC(C=1000,gamma=1).fit(X_train_scaled, y_train)
    train_scaled_score = svc.score(X_train_scaled, y_train)
    test_scaled_score = svc.score(X_test_scaled, y_test)
    print("Accuracy on training set: {:.3f}".format(train_scaled_score))    # 1.000
    print("Accuracy on test set: {:.3f}".format(test_scaled_score))         # 0.958






