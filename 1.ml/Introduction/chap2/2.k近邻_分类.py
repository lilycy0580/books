
# KNN分类
import mglearn
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

if __name__ == '__main__':
    # knn(k=1) forge
    mglearn.plots.plot_knn_classification(n_neighbors=1)
    plt.savefig("./../img/2.监督学习/4.单一最近邻模型对forge数据集的预测结果.png",dpi=1080)
    plt.show()

    # knn(k=3) forge
    mglearn.plots.plot_knn_classification(n_neighbors=3)
    plt.savefig("./../img/2.监督学习/5.3近邻模型对forge数据集的预测结果.png",dpi=1080)
    plt.show()

    # 模型训练 knn分类 forge k=3
    X, y = mglearn.datasets.make_forge()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn_clf = KNeighborsClassifier(n_neighbors=3)
    knn_clf.fit(X_train, y_train)
    predict = knn_clf.predict(X_test)
    score = knn_clf.score(X_test, y_test)
    print("Test set predictions:",predict)      # [1 0 1 0 1 0 0]
    print("Test set accuracy:",round(score,2))  # 0.86 在测试数据集中,模型对其中86%的样本预测的类别都是正确的

    # 绘制knn决策边界
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for n_neighbors, ax in zip([1, 3, 9], axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
        mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)   # 绘制决策边界
        mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)                            # 绘制散点图
        ax.set_title("{} neighbor(s)".format(n_neighbors))
        ax.set_xlabel("feature 0")
        ax.set_ylabel("feature 1")
    axes[0].legend(loc=3)
    plt.savefig("./../img/2.监督学习/6.不同n_neighbors值的k近邻模型的决策边界.png",dpi=1080)
    plt.show()
    """
    总结:
        用单一邻居绘制的决策边界紧跟着训练数据.随着邻居个数越来越多,决策边界也越来越平滑.更平滑的边界对应更简单的模型
        更少的邻居对应更高的模型复杂度,而使用更多的邻居对应更低的模型复杂度
    
    极端情况:    
        即邻居个数等于训练集中所有数据点的个数,那么每个测试点的邻居都完全相同(即所有训练点),所有预测结果也完全相同(即训练集中出现次数最多的类别)
    """

    # 分析KNeighborsClassifier
    # 模型复杂度与泛化能力的关系
    # 思想:先将数据集分成训练集和测试集,然后用不同的邻居个数对训练集和测试集的性能进行评估
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data,
                                                        cancer.target,
                                                        stratify=cancer.target,
                                                        random_state=66)
    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 11)       # [1,10]
    for n_neighbors in neighbors_settings:
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        test_score = clf.score(X_test, y_test)
        training_accuracy.append(train_score)
        test_accuracy.append(test_score)
    plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
    plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.savefig("./../img/2.监督学习/7.以n_neighbors为自变量,对比训练集精度和测试集精度.png",dpi=1080)
    plt.show()

    """
    总结:
        过拟合与欠拟合的一些特征:
            仅考虑单一近邻时,训练集上的预测结果十分完美.但随着邻居个数的增多,模型变得更简单,训练集精度也随之下降.
            单一邻居时的测试集精度比使用更多邻居时要低,这表示单一近邻的模型过于复杂.与之相反,当考虑10个邻居时,模型又过于简单,性能甚至变得更差.
        最佳性能在中间的某处,邻居个数大约为6.最差的性能约为88%的精度
    """


