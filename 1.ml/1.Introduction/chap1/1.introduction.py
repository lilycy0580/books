import mglearn
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    # 1.重要的库和工具
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x:{}".format(x))
    """
    [[1 2 3]
     [4 5 6]]
    """

    eye = np.eye(4)                 # 对角线元素为1,其余元素为0======> 单位矩阵/对角矩阵
    print("NumPy array:", eye)
    """
    [[1. 0. 0. 0.]
     [0. 1. 0. 0.]
     [0. 0. 1. 0.]
     [0. 0. 0. 1.]]
    """

    sparse_matrix = sparse.csr_matrix(eye)
    print("SciPy sparse CSR matrix:", sparse_matrix)
    """
    (0, 0)	1.0
    (1, 1)	1.0
    (2, 2)	1.0
    (3, 3)	1.0
    """

    data = np.ones(4)
    row_indices = np.arange(4)
    col_indices = np.arange(4)
    eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
    print("COO representation:", eye_coo)
    """
    (0, 0)	1.0
    (1, 1)	1.0
    (2, 2)	1.0
    (3, 3)	1.0
    """

    x = np.linspace(-10, 10, 100)       # [-10,10]区间共生成100个数
    y = np.sin(x)
    plt.plot(x, y, marker="x")
    plt.savefig("./img/1.png", dpi=1080)
    plt.show()

    data = {'Name': ["John", "Anna", "Peter", "Linda"],
            'Location': ["New York", "Paris", "Berlin", "London"],
            'Age': [24, 13, 53, 33]
            }
    data_pandas = pd.DataFrame(data)
    print(data_pandas,data_pandas[data_pandas.Age > 30])
    """
        Name  Location  Age
    0   John  New York   24
    1   Anna     Paris   13
    2  Peter    Berlin   53
    3  Linda    London   33

        Name Location  Age
    2  Peter   Berlin   53
    3  Linda   London   33
    """

    # 2.python Version
    import sys
    print("Python version:", sys.version)  # 书本:3.7.6
    import numpy as np
    print("NumPy version:", np.__version__)
    import matplotlib
    print("matplotlib version:", matplotlib.__version__)
    import scipy as sp
    print("SciPy version:", sp.__version__)
    import IPython
    print("IPython version:", IPython.__version__)
    import sklearn
    print("scikit-learn version:", sklearn.__version__)
    import pandas as pd
    print("pandas version:", pd.__version__)
    import nltk
    print("nltk version:", nltk.__version__)
    import spacy
    print("spacy version:", spacy.__version__)
    # pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.5.0/en_core_web_sm-3.5.0.tar.gz
    """
    Python version: 3.8.20 | packaged by conda-forge | (default, Sep 30 2024, 17:44:03) [MSC v.1929 64 bit (AMD64)]
    NumPy version: 1.23.5
    matplotlib version: 3.1.3
    SciPy version: 1.4.1
    IPython version: 8.12.2
    scikit-learn version: 1.0
    pandas version: 1.3.5
    nltk version: 3.8.1
    spacy version: 3.5.0
    """

    # 3.鸢尾花种类识别
    iris_dataset = load_iris()
    print("Keys of iris_dataset:", iris_dataset.keys())
    # print(iris_dataset['DESCR'][:193])
    print("Target names:", iris_dataset['target_names'])
    print("Feature names:", iris_dataset['feature_names'])
    print("Type of data:", type(iris_dataset['data']))
    print("Shape of data:", iris_dataset['data'].shape)
    print("First five rows of data:", iris_dataset['data'][:5])
    print("Type of target:", type(iris_dataset['target']))
    print("Shape of target:", iris_dataset['target'].shape)
    print("Target:", iris_dataset['target'])
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
    First five rows of data: 
        [[5.1 3.5 1.4 0.2]
         [4.9 3.  1.4 0.2]
         [4.7 3.2 1.3 0.2]
         [4.6 3.1 1.5 0.2]
         [5.  3.6 1.4 0.2]]
    Type of target: 
        <class 'numpy.ndarray'>
    Shape of target: 
        (150,)
    Target: 
        [0 0 0 0 ... 0 1 1... 1 1 2 2 ...2 2 2 2]
    """

    # 构建训练集与测试集 默认train:0.75 test:0.25
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'],
                                                        iris_dataset['target'],
                                                        train_size=0.75,                 # 指定训练集占75%,即默认值
                                                        random_state=0)
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print(X_train[0])
    """
    X_train shape: (112, 4)
    y_train shape: (112,)
    X_test shape: (38, 4)
    y_test shape: (38,)
    [5.9 3.  4.2 1.5]   对应iris_dataset.feature_names 即sepal_length,sepal_width,petal_length,petal_width(cm)
    """

    # 数据可视化(散点图) 发现异常值与特殊值
    # 两两查看所有的特征,矩阵的对角线是每个特征的直方图
    # 从图中可看出利用花瓣和花萼的测量数据基本可以将三个类别区分开
    iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
    pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15),
                               marker='o', hist_kwds={'bins': 20}, s=60,
                               alpha=.8, cmap=mglearn.cm3)
    plt.savefig('./img/2.png',dpi=1080)
    plt.show()

    # 4.第一个KNN模型
    knn = KNeighborsClassifier(n_neighbors=1)       # 1.构建模型:K近邻算法
    knn.fit(X_train, y_train)                       # 2.训练数据

    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = knn.predict(X_new)                 # 3.预测新数据
    print("X_new.shape:", X_new.shape)
    print("Prediction:", prediction)
    print("Predicted target name:",iris_dataset['target_names'][prediction])
    """
    X_new.shape: (1, 4)
    Prediction: [0]
    Predicted target name: ['setosa']
    """

    # 模型评估
    # 方式一:计算精度
    y_pred = knn.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    # 方式二:
    score = knn.score(X_test, y_test)
    print("Test set predictions:", y_pred)
    print("Test set score: {:.2f}".format(accuracy))
    print("Test set score: {:.2f}".format(score))

    # 总结:模型训练+评估
    X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    print("Test set score: {:.2f}".format(score))
    """
    Test set predictions: [2 1 0 2 0 2 0 1 1 1 2 1 1 1 1 0 1 1 0 0 2 1 0 0 2 0 0 1 1 0 2 1 0 2 2 1 0 2]
    Test set score: 0.97
    Test set score: 0.97
    Test set score: 0.97
    """






