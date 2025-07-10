import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """
    梯度提升回归树:
        采用连续的方式构造树,每棵树都试图纠正前一棵树的错误
        默认情况下,梯度提升回归树中没有随机化,而是用到了强预剪枝 深度为1到5之间的树
        对参数设置更加敏感,是机器学习竞赛的优胜者
    """
    # 100棵树 最大深度为3,学习率为0.1
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    gbrt = GradientBoostingClassifier(random_state=0)
    gbrt.fit(X_train, y_train)
    train_score = gbrt.score(X_train, y_train)
    test_score = gbrt.score(X_test, y_test)
    print("Accuracy on training set: {:.3f}".format(train_score))   # 1.000   过拟合
    print("Accuracy on test set: {:.3f}".format(test_score))        # 0.965

    # 降低过拟合,使用预剪枝 限制最大深度/降低学习率 =======> 降低模型复杂度
    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)
    train_score = gbrt.score(X_train, y_train)
    test_score = gbrt.score(X_test, y_test)
    print("Accuracy on training set: {:.3f}".format(train_score))   # 0.991  降低训练集精度
    print("Accuracy on test set: {:.3f}".format(test_score))        # 0.972  模型性能显著提高

    gbrt = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
    gbrt.fit(X_train, y_train)
    train_score = gbrt.score(X_train, y_train)
    test_score = gbrt.score(X_test, y_test)
    print("Accuracy on training set: {:.3f}".format(train_score))   # 0.988
    print("Accuracy on test set: {:.3f}".format(test_score))        # 0.965 模型泛化能力没有显著提高

    # 特征重要性
    def plot_feature_importances_cancer(model):
        n_features = cancer.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        plt.tight_layout()
        plt.savefig("./../img/2.监督学习/35.用于拟合乳腺癌数据集的梯度提升分类器给出的特征重要性.png", dpi=1080)
        plt.show()

    gbrt = GradientBoostingClassifier(random_state=0, max_depth=1)
    gbrt.fit(X_train, y_train)
    plot_feature_importances_cancer(gbrt)
    # 总结:梯度提升树的特征重要性与随机森林的特征重要性有些类似,不过梯度提升完全忽略了某些特征

    """
    随机森林 vs 梯度提升树
        先尝试随机森林,它鲁棒性很好.若随机森林效果很好但预测时间太长,或者机器学习模型精度小数点后第二位的提高也很重要,则切换成梯度提升通常会有用
    """

