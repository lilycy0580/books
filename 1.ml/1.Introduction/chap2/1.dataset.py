# 控制台不输出warn警告信息
import warnings
warnings.filterwarnings('ignore')

import mglearn
from matplotlib import pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_boston

if __name__ == '__main__':
    """
    模拟数据集:
        forge   二分类     (26, 2)
        wave    回归       (40, 1)
        
    真实数据集:
        cancer  二分类     (569, 30) 
        boston  回归       (506, 13)
        extended_boston   (506, 104)
    """
    # forge 二分类
    X, y = mglearn.datasets.make_forge()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)       # (26, 2) (26,)
    plt.legend(["Class 0", "Class 1"], loc=4)
    plt.xlabel("First feature")
    plt.ylabel("Second feature")
    plt.savefig("./img/2.forge数据集的散点图.png",dpi=1080)
    plt.show()

    # wave 回归算法
    X, y = mglearn.datasets.make_wave(n_samples=40)     # (40, 1) (40,)
    plt.plot(X, y, 'o')
    plt.ylim(-3, 3)
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.savefig("./img/3.wave数据集的图像,x轴表示特征,y轴表示回归目标.png",dpi=1080)
    plt.show()

    # 癌症数据集
    cancer = load_breast_cancer()
    print("cancer.keys:", cancer.keys())
    print("Shape of cancer data:", cancer.data.shape)   # (569, 30)
    dict = {n:v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
    print("Sample counts per class:",dict)              # {'malignant': 212, 'benign': 357}
    print("Feature names:", cancer.feature_names)       # etc

    # 波士顿房价
    boston = load_boston()
    print("Data shape:", boston.data.shape)             # (506, 13)
    X, y = mglearn.datasets.load_extended_boston()      # (506, 104) 扩展数据集 输入特征=原始特征及其交互项(交互项:特征之间的乘积)
    print("X.shape:", X.shape)                          # [a  a^2  ab]   13+13+(13*12)/2=104

    from sklearn.datasets import fetch_california_housing
    housing = fetch_california_housing()
    print(housing.data.shape)

    import warnings
    warnings.filterwarnings('ignore')


