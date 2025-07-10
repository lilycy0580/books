
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
    """
    决策树回归:
        DecisionTreeRegressor 基于树的模型用于回归时,不能外推,也不能在训练数据范围之外进行预测
    """
    # RAM历史价格数据集 读取数据构建数据集
    ram_prices = pd.read_csv(os.path.join(mglearn.datasets.DATA_PATH, "./../data/ram_price.csv"))
    plt.semilogy(ram_prices.date, ram_prices.price)  # y轴取对数
    plt.xlabel("Year")
    plt.ylabel("Price in $/Mbyte")
    plt.savefig("./../img/2.监督学习/31.用对数坐标绘制RAM价格的历史发展.png", dpi=1080)
    plt.show()

    data_train = ram_prices[ram_prices.date < 2000]  # (202, 3)
    data_test = ram_prices[ram_prices.date >= 2000]  # (131, 3)
    X_train = data_train.date[:, np.newaxis]  # (202,) ==> (202, 1) 给数组增加一个维度
    y_train = np.log(data_train.price)

    tree_reg = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)  # 决策树 vs 线性回归
    linear_reg = LinearRegression().fit(X_train, y_train)

    X_all = ram_prices.date[:, np.newaxis]
    pred_tree = tree_reg.predict(X_all)
    pred_lr = linear_reg.predict(X_all)

    price_tree = np.exp(pred_tree)
    price_lr = np.exp(pred_lr)

    plt.semilogy(data_train.date, data_train.price, label="Training data")
    plt.semilogy(data_test.date, data_test.price, label="Test data")
    plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
    plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
    plt.legend()
    plt.savefig("./../img/2.监督学习/32.线性模型和回归树对RAM价格数据的预测结果对比.png", dpi=1080)
    plt.show()

    """
    回归:
        线性模型
            线性模型用一条直线对数据做近似,对测试集预测较好,但忽略训练集和测试集中一些细微的变化
    
        决策树
            完美预测测试集,由于未限制树的复杂度,故记住整个数据集,一旦输入超出模型训练数据的范围,模型只能维持预测最后一个已知数据点
            无法在训练数据的范围之外生成新的响应,所有基于树的模型都有这个缺点 
    """



