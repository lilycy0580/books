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
    """
    Keys of iris_dataset: 
        dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
    Target names: 
        ['setosa' 'versicolor' 'virginica']
    Feature names: 
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    Type of data: 
        <class 'numpy.ndarray'>
    Shape of data: 
        (150, 4)
    Type of target: 
        <class 'numpy.ndarray'>
    Shape of target: 
        (150,)
    First five rows of data:
         [[5.1 3.5 1.4 0.2]
         [4.9 3.  1.4 0.2]
         [4.7 3.2 1.3 0.2]
         [4.6 3.1 1.5 0.2]
         [5.  3.6 1.4 0.2]]
    Target:
     [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 
     1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
     2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2]
    """

    # 2.衡量模型是否成功:训练数据与测试数据 75%/25%
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'],
                                                        random_state=0)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    """
    X_train shape: (112, 4)
    y_train shape: (112,)
    X_test shape: (38, 4)
    y_test shape: (38,)
    """

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
    plt.savefig('./img/2.Iris数据集的散点图矩阵,按类别标签着色.png',dpi=1080)
    plt.show()

    # 4.构建模型:K近邻算法
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # 5.对新数据做出预测
    X_new = np.array([[5, 2.9, 1, 0.2]])    # shape (1, 4)
    prediction = knn.predict(X_new)
    print("Prediction:", prediction)        # [0]
    print("Predicted target name:",iris_dataset['target_names'][prediction]) # 'setosa'

    # 6.评估模型 指标:精度
    # 方式一:
    y_pred = knn.predict(X_test)            # 测试集
    accuracy = np.mean(y_pred == y_test)    # 计算分类模型准确率 计算精度,即平均值
    # 方式二:
    knn_score = knn.score(X_test, y_test)   # 计算测试集的精度 对于测试集中的鸢尾花,我们的预测有97%是正确的
    print("Test set predictions:", y_pred)
    print("Test set score: {:.2f}".format(accuracy))
    print("Test set score: {:.2f}".format(knn_score))   # 0.97

    # 总结:模型训练+评估
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'],
                                                        random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    accuracy = knn.score(X_test, y_test)
    print("Test set score: {:.2f}".format(accuracy))    # 0.97
