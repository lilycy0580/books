import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    """
    随机森林:
        决策树的缺点为对训练集过拟合,随机森林本质上是决策树的集合,每棵树与其他树都不同
        
        思想:
            每棵树预测都可能相对较好,但可能对部分数据过拟合.
            构造多棵树且每棵树的预测都很好,但均以不同的方式过拟合,则对树的结果取平均值来降低过拟合,既能减少过拟合又能保持树的预测能力
                             
        随机化:
            1.选择用于构造树的数据点
            2.选择每次划分测试的特征
    """

    # 1.构造随机森林
    """
    构造随机森林:
        1.确定用于构造的树的个数 
            对数据进行自助采样 从n_samples个数据点中有放回地重复随机抽取一个样本,共抽取n_samples次 
            决策树个数  n_estimators
            
        2.构造决策树
            在每个结点处,算法随机选择特征的一个子集,并对其中一个特征寻找最佳测试,而不是对每个结点都寻找最佳测试
            每个结点中特征子集的选择是相互独立
            特征个数 max_features
        
        保证随机森林中所有树都不相同:
            自助采用保证随机森林中每棵决策树的数据集略有不同
            每个结点的特征选择,每棵树的每次划分都是基于特征的不同子集
            
        max_features
            较大 随机森林中的树十分相似,利用最独特的特征可轻松拟合数据
            较小 随机森林中的树会差异很大,为更好拟合数据,每棵树的深度要很大 
            
    随机森林预测:
        分类 每个树给出每个可能的输出标签的概率,对所有树的预测概率取平均值,然后将概率最大的类别作为预测结果
        回归 每个树进行预测,将结果取平均值作为最终预测结果
    """

    # 2.分析随机森林
    # 5棵树构造的随机森林,构造make_moons数据集
    X, y = make_moons(n_samples=100, noise=0.25, random_state=3)    # (100, 2) (100,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
    forest = RandomForestClassifier(n_estimators=5, random_state=2).fit(X_train, y_train)

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))    # 2×3 子图
    for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
        ax.set_title("Tree {}".format(i))
        mglearn.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
    mglearn.plots.plot_2d_separator(forest,
                                    X_train,
                                    fill=True,
                                    ax=axes[-1, -1],
                                    alpha=.4)
    mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    axes[-1, -1].set_title("Random Forest")
    plt.savefig("./../img/2.监督学习/33.5棵随机化的决策树找到的决策边界,以及将它们的预测概率取平均后得到的决策边界.png",dpi=1080)
    plt.tight_layout()
    plt.show()
    """
    总结:
        5棵树学到的决策边界大不相同,每棵树都犯了一些错误,因为自助采样,一些训练点实际上并没有包含在这些树的训练集中
        随机森林比单独每一棵树的过拟合都要小,给出的决策边界也更符合直觉
        更多棵树(通常是几百或上千),从而得到更平滑的边界
    """

    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    forest = RandomForestClassifier(n_estimators=100, random_state=0)
    forest.fit(X_train, y_train)
    train_score = forest.score(X_train, y_train)
    test_score = forest.score(X_test, y_test)
    print("Accuracy on training set: {:.3f}".format(train_score))   # 1.000
    print("Accuracy on test set: {:.3f}".format(test_score))        # 0.972
    # 总结:未调节参数情况下,随机森林的精度为97% ,性能优于线性模型或单棵决策树

    # 特征重要性
    def plot_feature_importances_cancer(model):
        n_features = cancer.data.shape[1]
        plt.barh(np.arange(n_features), model.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel("Feature importance")
        plt.ylabel("Feature")
        plt.ylim(-1, n_features)
        plt.tight_layout()
        plt.savefig("./../img/2.监督学习/34.拟合乳腺癌数据集得到的随机森林的特征重要性.png", dpi=1080)
        plt.show()

    plot_feature_importances_cancer(forest)
    """
    特征重要性:
        随机森林中所有树的特征重要性求和并取平均(一般来说,随机森林给出的特征重要性要比单棵树给出的更为可靠)
        
    总结:    
        与单棵树相比随机森林中有更多特征的重要性不为零    
        类似决策树,随机森林也给出worst radius特征很大的重要性,但却选择worst perimeter作为信息量最大的特征
        
        由于构造随机森林过程中的随机性,随机森林比单棵树更能从总体把握数据的特征
    """




