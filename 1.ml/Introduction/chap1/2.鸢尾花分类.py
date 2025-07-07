import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 鸢尾花分类

if __name__ == '__main__':
    # 1.初识数据
    iris_dataset = load_iris()
    print("Keys of iris_dataset:", iris_dataset.keys())
    print("Target names:", iris_dataset['target_names'])
    print("Feature names:", iris_dataset['feature_names'])
    print("Type of data:", type(iris_dataset['data']))
    print("Shape of data:", iris_dataset['data'].shape)
    print("Type of target:", type(iris_dataset['target']))
    print("Shape of target:", iris_dataset['target'].shape)
    print("First five rows of data:\n", iris_dataset['data'][:5])
    print("Target:\n", iris_dataset['target'])

    # 2.衡量模型是否成功:训练数据与测试数据 75%/25%
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'],
                                                        random_state=0)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    # 3.观察数据===>数据可视化(散点图) 发现异常值与特殊值
    # 两两查看所有的特征,矩阵的对角线是每个特征的直方图
    # 从图中可看出利用花瓣和花萼的测量数据基本可以将三个类别区分开
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    pd.plotting.scatter_matrix(iris_dataframe,
                               c=y_train,
                               figsize=(15, 15),
                               marker='o',
                               hist_kwds={'bins': 20},
                               s=60,
                               alpha=0.8,
                               cmap=mglearn.cm3)
    plt.savefig('./../img/1.引言/2.Iris数据集的散点图矩阵,按类别标签着色.png')
    plt.show()

    # 4.构建模型:K近邻算法
    # 对一个新数据进行预测,模型在训练集中寻找与这个数据点距离最近的数据点,将其标签赋值给这个新数据点
    # k 训练集中与新数据点最近的任意k个邻居,用邻居中数量最多的类别做出预测
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # 5.做出预测
    X_new = np.array([[5, 2.9, 1, 0.2]])    # shape (1, 4)
    prediction = knn.predict(X_new)
    print("Prediction:", prediction)        # [0]
    print("Predicted target name:",iris_dataset['target_names'][prediction]) # 'setosa'

    # 6.评估模型
    y_pred = knn.predict(X_test)            # 测试集
    knn_score = knn.score(X_test, y_test)   # 计算测试集的精度 对于测试集中的鸢尾花,我们的预测有97%是正确的
    accuracy = np.mean(y_pred == y_test)    # 计算分类模型准确率
    # accuracy = (y_pred == y_test).mean()
    print("Test set predictions:", y_pred)
    print("Test set score: {:.2f}".format(knn_score))   # 0.97
    print("Test set score: {:.2f}".format(accuracy))

    # 总结:模型训练+评估
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'],
                                                        random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
