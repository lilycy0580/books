import os
import numpy as np
import pandas as pd
import mglearn
from matplotlib._pylab_helpers import Gcf
from mglearn.plot_interactive_tree import tree_image
from mglearn.plot_interactive_tree import plot_tree_partition
from sklearn.datasets import make_moons, load_breast_cancer, make_blobs
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz

if __name__ == '__main__':
    # 决策树分类
    # 区分4种动物  熊,鹰,企鹅,海豚
    mglearn.plots.plot_animal_tree()
    plt.savefig("./../img/2.监督学习/22.区分几种动物的决策树.png",dpi=1080)
    plt.show()
    """
    总结:
        树的非叶结点代表一个问题,叶节点代表一个答案 边将问题的答案与下一个问题连接起来

    机器学习语言:
        为区分4种动物,使用三个特征(羽毛,飞,鳍)构建一个模型,利用监督学习从数据中学习模型,无需人为构建模型
    """

    # 1.构建决策树
    """
    数据集:
        two_moons 两个半月 50个数据点

    测试与数据:
        学习决策树就是学习一些系列if/else问题,以便快速得到答案  这些问题称为测试 每个测试仅关注一个特征
        数据表示为连续特征,用于连续数据的测试形势为:特征i的值是否大于a?

    分类:
        构造决策树:
            算法搜遍所有可能的测试,找出对目标变量来说信息量最大的那一个,对数据反复递归划分,直到叶结点是纯的
            叶结点仅包含单一目标值(单一类别或单一回归值) 纯的

        预测新数据:
            查看该点位于特征空间划分的哪个趋于,将该区域的多数目标值(若叶节点是纯的,则为单一目标值)作为预测结果
            从根结点开始对树进行遍历就可以找到这一区域,每一步向左还是向右取决于是否满足相应的测试

    回归:
        构造决策树 略
        预测新数据
            基于每个结点的测试对树进行遍历,最终找到新数据点所属的叶结点.这一数据点的输出即为此叶结点中所有训练点的平均目标值
    """

    # 数据集 two_moons
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend(["Class 0", "Class 1"], loc='best')
    plt.savefig("./../img/2.监督学习/23.用于构造决策树的two_moons数据集.png",dpi=1080)
    plt.show()

    # 绘制决策边界与相应的树
    figs = []
    axes = []
    for i in range(3):
        fig, ax = plt.subplots(1, 2,  # 创建子图,每张图有2个子图
                               figsize=(12, 4),  # 设置每个子图不显示x轴与y轴刻度
                               subplot_kw={'xticks': (), 'yticks': ()})
        axes.append(ax)
    axes = np.array(axes)  # <class 'list'> => <class 'numpy.ndarray'> 将当前创建的坐标轴对象添加到axes列表中 (3, 2)
    for i, max_depth in enumerate([1, 2, 9]):
        tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(X, y)
        ax = plot_tree_partition(X, y, tree, ax=axes[i, 0])
        ax.set_title("depth = %d" % max_depth)
        img = tree_image(tree)
        axes[i, 1].imshow(img)
        axes[i, 1].set_axis_off()
    # 获取所有图形管理器,从管理器获取图形对象并保存
    managers = Gcf.get_all_fig_managers()
    for i, manager in enumerate(managers, 1):
        fig = manager.canvas.figure
        fig.savefig("./../img/2.监督学习/{}.深度为{}的树的决策边界(左)与相应的树(右).png".format(23 + i, i), dpi=1080)
    plt.show()

    # 2.控制决策树的复杂度
    """
    总结:
        构造决策树直到所有叶结点都是纯的叶结点,会导致模型非常复杂并对训练数据高度过拟合
        (纯叶结点的存在说明这棵树在训练集上的精度是100%)

    防止过拟合:
        预剪枝 及早停止树的生长 括限制树的最大深度,限制叶结点的最大数目,规定一个结点中数据点的最小数目
        后剪枝/剪枝 先构造树随后删除或折叠信息量很少的结点

    决策树:(sklearn仅实现预剪枝)
        DecisionTreeClassifier
        DecisionTreeRegressor
    """

    # DTC + cancer
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
    tree = DecisionTreeClassifier(random_state=0)
    tree.fit(X_train, y_train)
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    print("Accuracy on training set: {:.3f}".format(train_score))   # 1.000 训练集精度为100%(叶结点为纯的)
    print("Accuracy on test set: {:.3f}".format(test_score))        # 0.937 测试集精度低于线性模型 线性模型为95%
    # 总结:未剪枝的决策树容易过拟合,对新数据泛化能力不佳

    tree = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree.fit(X_train, y_train)
    train_score = tree.score(X_train, y_train)
    test_score = tree.score(X_test, y_test)
    print("Accuracy on training set: {:.3f}".format(train_score))   # 0.988
    print("Accuracy on test set: {:.3f}".format(test_score))        # 0.951
    """
    预剪枝:
        在决策树上使用预剪枝可在完美拟合训练集之前阻止树的展开

        方案:
            限制树的深度,降低训练集精度,提高测试集精度(达到一定深度后停止树的展开 max_depth=4)
    """

    # 3.分析决策树
    # 将树可视化,向非机器学习专家解释 树的深度为4,有点太大
    export_graphviz(tree,
                    out_file="tree.dot",
                    class_names=["malignant", "benign"],
                    feature_names=cancer.feature_names,
                    impurity=False,
                    filled=True)
    with open("tree.dot") as f:
        dot_graph = f.read()
    graph = graphviz.Source(dot_graph)
    graph.render("./../img/2.监督学习/27.基于乳腺癌数据集构造的决策树的可视化", format='png', cleanup=True)

    # 4.树的特征重要性
    def plot_feature_importances_cancer(model):
        n_features = cancer.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        plt.tight_layout()
        plt.savefig("./../img/2.监督学习/28.在乳腺癌数据集上学到的决策树的特征重要性.png", dpi=1080)
        plt.show()

    print("Feature importances:",tree.feature_importances_)  # sum = 1
    plot_feature_importances_cancer(tree)        # 特征重要性:worst radius为最重要的特征 即决策树第一层划分已经将两个类别区分得很好

    """
    特征重要性:
        对每个特征对树的决策的重要性进行排序 值在[0,1]之间,0表示根本没用到,1表示完美预测目标值 特征重要性的求和始终为1
        若某个特征的特征重要性比较小,不能说明其没有提供任何信息,仅能说明该信息没有被树选中,可能是另一个特征也包含相同的信息
        特征重要性仅反应出某个特征很重要,但对应的值不会映射出对应的样本类别
    """
    X, y = make_blobs(centers=4, random_state=8)    # (100, 2) (100,) 原始4簇,手动变成2类别
    y = y % 2
    tree = DecisionTreeClassifier(random_state=0).fit(X, y)
    plt.figure()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    mglearn.plots.plot_2d_separator(tree, X, linestyle="dashed")
    plt.legend(["Class 0", "Class 1"], loc="best")
    plt.savefig("./../img/2.监督学习/29.一个二维数据集(y轴上的特征与类别标签是非单调的关系)与决策树给出的决策边界.png", dpi=1080)
    plt.show()

    # 决策树可视化
    export_graphviz(tree, out_file="mytree.dot", impurity=False, filled=True)
    with open("mytree.dot") as f:
        dot_graph = f.read()
    print("Feature importances:",tree.feature_importances_) # [0. 1.]
    graph = graphviz.Source(dot_graph)
    graph.render("./../img/2.监督学习/30.从图2-29的数据中学到的决策树", format='png', cleanup=True)
    # 总结:所有信息均包含在第二个特征中 较大的X[1]对应类别0,较小的X[1]对应类别1 ×



