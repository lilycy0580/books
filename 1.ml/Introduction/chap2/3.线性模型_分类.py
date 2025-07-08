
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import make_blobs

if __name__ == '__main__':
    # 1.二分类
    # 逻辑回归 vs 线性支持向量机 LinearSVC  forge
    X, y = mglearn.datasets.make_forge()    # (26, 2) (26,)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
        clf = model.fit(X, y)
        mglearn.plots.plot_2d_separator(clf,
                                        X,
                                        fill=False,
                                        eps=0.5,
                                        ax=ax,
                                        alpha=0.7)
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title(clf.__class__.__name__)
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    axes[0].legend()
    plt.savefig("./../img/2.监督学习/15.线性SVM和Logistic回归在forge数据集上的决策边界(均为默认参数).png",dpi=1080)
    plt.show()

    """
    总结:
        两个模型决策边界都是直线,上面为类别1,下面为类别0 
        两模型均使用L2正则化,决定正则化强度的参数为C. 
            C值越大,对应的正则化越弱,训练集拟合到最好,模型更强调每个数据点都分类正确的重要性
            C值越小,模型更强调使系数向量w接近于0,模型近邻适应"大多数"数据点
    """

    # LinearSVC 正则化系数c
    mglearn.plots.plot_linear_svc_regularization()
    plt.savefig("./../img/2.监督学习/16.不同C值的线性SVM在forge数据集上的决策边界.png",dpi=1080)
    plt.show()

    """
    正则化系数C:
        C值较小,对应强正则化模型,决策边界会选择相对水平的线 有两个点分类错误
        C值稍大,模型更光柱两个分类错误的样本,使决策边界斜率变大
        C值非常大,决策边界的斜率也很大,对类别0中所有点分类正确,类别1中有一个点分类错误 模型可能过拟合
        
    分类的线性模型在低维空间可能受限,决策边界可能为直线或平面,但在高维空间,分类线性模型十分强大,考虑更多特征时需注意避免过拟合
    """

    # 逻辑回归 cancer
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    logreg001 = LogisticRegression(C=0.01,max_iter=2000).fit(X_train, y_train)
    train_score = logreg001.score(X_train, y_train)
    test_score = logreg001.score(X_test, y_test)
    print("Training set score: {:.3f}".format(train_score))             # 0.953
    print("Test set score: {:.3f}".format(test_score))                  # 0.951 正则化更强,模型更加欠拟合,训练集和测试机精度下降

    logreg = LogisticRegression(max_iter=2000).fit(X_train, y_train)  # C=1 默认值
    train_score = logreg.score(X_train, y_train)
    test_score = logreg.score(X_test, y_test)
    print("Training set score: {:.3f}".format(train_score))             # 0.958
    print("Test set score: {:.3f}".format(test_score))                  # 0.958 精度达到95%,训练集与测试集性能接近,可能欠拟合

    logreg100 = LogisticRegression(C=100,max_iter=2000).fit(X_train, y_train)
    train_score = logreg100.score(X_train, y_train)
    test_score = logreg100.score(X_test, y_test)
    print("Training set score: {:.3f}".format(train_score))             # 0.981
    print("Test set score: {:.3f}".format(test_score))                  # 0.972 训练集与测试集精度均提高 即更复杂的模型性能更好

    # L2正则化 不同正则化参数C对模型的影响
    plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
    plt.plot(logreg.coef_.T, 'o', label="C=1")
    plt.plot(logreg100.coef_.T, '^', label="C=100")
    plt.xticks(range(cancer.data.shape[1]),  # 获取data的特征数,作为x轴刻度标签个数
               cancer.feature_names,         # 特征的名称列表作为x轴刻度标签
               rotation=90)                  # 刻度标签旋转90度
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])     # 基准线 y=0 x=[xlims[0],xlims[1]]
    plt.xlim(xlims)
    plt.ylim(-5, 5)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.legend()
    plt.tight_layout()                       # 自动调整布局
    plt.savefig("./../img/2.监督学习/17.不同C值的Logistic回归在乳腺癌数据集上学到的系数.png",dpi=1080)
    plt.show()

    """
    总结:
        逻辑回归默认L2正则化,性能与lasso类似 
        更强的正则化使系数更趋于0,但系数不会正好等于0
        
        想要一个可解释性更强的模型,L1正则化可能更合适,约束模型只使用少数几个特征
    """

    # L1正则化
    for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
        lr_l1 = LogisticRegression(C=C,
                                   max_iter = 2000,
                                   solver='liblinear',
                                   penalty="l1")    # 惩罚项,影响正则化,也会影响模型是使用全部特征还是使用特征的子集
        lr_l1.fit(X_train, y_train)
        train_score = lr_l1.score(X_train, y_train)
        test_score = lr_l1.score(X_test, y_test)
        print("Training accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, train_score)) # 0.91  0.96  0.99
        print("Test accuracy of l1 logreg with C={:.3f}: {:.2f}".format(C, test_score))      # 0.92  0.96  0.98
        plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
    plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)  # x轴刻度
    xlims = plt.xlim()
    plt.hlines(0, xlims[0], xlims[1])
    plt.xlim(xlims)
    plt.xlabel("Feature")
    plt.ylabel("Coefficient magnitude")
    plt.ylim(-5, 5)
    plt.legend(loc=3)
    plt.tight_layout()                       # 自动调整布局
    plt.savefig("./../img/2.监督学习/18.对于不同的C值,L1惩罚的逻辑回归在乳腺癌数据集上学到的系数.png",dpi=1080)
    plt.show()

    # 2.多分类
    """
        许多线性分类模型只适用于二分类问题,不能推广到多类别问题,逻辑回归除外
    
    多分类:    
        每个类别都对应一个二类分类器,这样每个类别也都有一个系数向量w和一个截距b,其结果中最大值对应的类别即为预测的类别标签
        w[0] * x[0] + w[1] * x[1] + … + w[p] * x[p] + b
        
    """
    # 三分类数据集,每个类别的数据都是从一个高斯分布中采样得出
    X, y = make_blobs(random_state=42)  # (100, 2) (100,)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(["Class 0", "Class 1", "Class 2"])
    plt.savefig("./../img/2.监督学习/19.包含3个类别的二维玩具数据集.png",dpi=1080)
    plt.show()

    # LinearSVC分类器 训练模型
    linear_svm = LinearSVC().fit(X, y)
    print("Coefficient shape: ", linear_svm.coef_.shape)        # 斜率 w  (3, 2)
    print("Intercept shape: ", linear_svm.intercept_.shape)     # 截距 b  (3,)

    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15, 15)
    for coef, intercept, color in zip(linear_svm.coef_,linear_svm.intercept_,mglearn.cm3.colors):
        # 绘制决策边界
        plt.plot(line,                              # x轴上的值范围
                 -(line * coef[0] + intercept) / coef[1], # -(w[0]*x+b)/w[1]    w[0]*x + w[1]*y + b = 0
                 c=color)                                 # 指定线条颜色
    plt.ylim(-10, 15)
    plt.xlim(-10, 8)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
    plt.tight_layout()
    plt.savefig("./../img/2.监督学习/20.三个“一对其余”分类器学到的决策边界.png",dpi=1080)
    plt.show()

    """
    总结:
        类别0的分类器的分类置信方程的结果大于0,其他两个类别对应的结果都小于0
            类别0的点位于类别0的直线上方 被类别0的二类分类器划分为"类别0"
            类别0的点位于类别2的直线上方 被类别2的二类分类器划分为"其余"
            类别0的点位于类别1的直线左侧 被类别1的二类分类器划分为"其余"
        
        三角形区域的类别属于分类方程结果最大的那个类别,即最接近的那条线对应的类别
    """
    # 三角区域类别划分
    mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=0.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    line = np.linspace(-15, 15)
    for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, mglearn.cm3.colors):
        plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1', 'Line class 2'], loc=(1.01, 0.3))
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.tight_layout()
    plt.savefig("./../img/2.监督学习/21.三个“一对其余”分类器得到的多分类决策边界.png",dpi=1080)
    plt.show()