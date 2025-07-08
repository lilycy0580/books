
# KNN回归
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

if __name__ == '__main__':
    # knn回归
    # knn(k=1) wave
    mglearn.plots.plot_knn_regression(n_neighbors=1)
    plt.savefig("./../img/2.监督学习/8.单一近邻回归对wave数据集的预测结果.png",dpi=1080)
    plt.show()

    # knn(k=3) wave
    mglearn.plots.plot_knn_regression(n_neighbors=3)
    plt.savefig("./../img/2.监督学习/9.3个近邻回归对wave数据集的预测结果.png",dpi=1080)
    plt.show()

    # 模型训练 knn回归 wave k=3
    X, y = mglearn.datasets.make_wave(n_samples=40) # X:(40, 1) y:(40,)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn_reg = KNeighborsRegressor(n_neighbors=3)
    knn_reg.fit(X_train, y_train)
    predict = knn_reg.predict(X_test)           # 对测试集进行预测
    score = knn_reg.score(X_test, y_test)       # 评估模型  0.83,模型的拟合相对较好
    print("Test set predictions:",predict)
    print("Test set R^2:",np.round(score,2))

    """
    R^2:
        回归模型预测的优度度量,位于0到1之间 
        1:完美预测
        0:常数模型,即总是预测训练集响应(y_train)的平均值
    """

    # 分析KNeighborsRegressor
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    line = np.linspace(-3, 3, 1000).reshape(-1, 1)  # (1000, 1)
    for n_neighbors, ax in zip([1, 3, 9], axes):
        reg = KNeighborsRegressor(n_neighbors=n_neighbors)
        reg.fit(X_train, y_train)
        predict = reg.predict(line)
        train_score = reg.score(X_train, y_train)
        test_score = reg.score(X_test, y_test)

        ax.plot(line,predict)
        ax.plot(X_train, y_train, '^', c=mglearn.cm2(0), markersize=8)
        ax.plot(X_test, y_test, 'v', c=mglearn.cm2(1), markersize=8)
        ax.set_title("{} neighbor(s) train score: {:.2f} test score: {:.2f}".format(n_neighbors, train_score,test_score))
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
    axes[0].legend(["Model predictions", "Training data/target","Test data/target"], loc="best")
    plt.savefig("./../img/2.监督学习/10.不同n_neighbors值的k近邻回归的预测结果对比.png",dpi=1080)
    plt.show()

    """
    总结:
        仅使用单一邻居,训练集中的每个点都对预测结果有显著影响.预测结果的图像经过所有数据点,这导致预测结果非常不稳定
        考虑更多的邻居之后,预测结果变得更加平滑.但对训练数据的拟合也不好
    """







