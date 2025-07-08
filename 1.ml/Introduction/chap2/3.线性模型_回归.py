
import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

if __name__ == '__main__':
    # 1.用于回归的线性模型
    mglearn.plots.plot_linear_regression_wave() # w[0]: 0.393906  b: -0.031804
    plt.savefig("./../img/2.监督学习/11.线性模型对wave数据集的预测结果.png",dpi=1080)
    plt.show()

    """
    回归模型:
        对单一特征的预测结果是一条直线.两个特征时是一个平面.或者在更高维度(即更多特征)时是一个超平面
        
        特点:
            适用于有多个特征的数据集
            若特征数量大于训练集数据的数量,则任何目标y可以在训练集上用线性函数完美拟合
        
        多种回归模型区别:
            如何从训练集中学习参数w和b
            控制模型复杂度
    """

    # 2.线性回归  最小二乘法   wave+boston
    X, y = mglearn.datasets.make_wave(n_samples=60)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    lr = LinearRegression().fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print("lr.coef_:", lr.coef_)            # 斜率 w
    print("lr.intercept_:", lr.intercept_)  # 偏移/截距 b
    print("Training set score: {:.2f}".format(train_score)) # 0.67
    print("Test set score: {:.2f}".format(test_score))      # 0.66  R^2

    # 总结:训练集和测试集上的分数非常接近,拟合效果不好 可能存在欠拟合

    X, y = mglearn.datasets.load_extended_boston()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0) # X_train:(379, 104) y_test:(127,)
    lr = LinearRegression().fit(X_train, y_train)
    train_score = lr.score(X_train, y_train)
    test_score = lr.score(X_test, y_test)
    print("Training set score: {:.2f}".format(train_score)) # 0.95
    print("Test set score: {:.2f}".format(test_score))      # 0.61  R^2

    # 总结:在训练集上的预测非常准确,测试集的R^2要低很多 过拟合
    # 模拟数据欠拟合,真实数据过拟合,故需要找到可以控制复杂度的模型----->岭回归

    """
    线性回归:
        寻找参数w和b,使得对训练集的预测值与真实的回归目标值y之间的均方误差最小
        均方误差 MSE 是预测值与真实值之差的平方和除以样本数,线性回归没有参数,无法控制模型的复杂度
    """

    # 3.岭回归  boston   L2正则化
    ridge01 = Ridge(alpha=0.1).fit(X_train, y_train)
    train_score = ridge01.score(X_train, y_train)
    test_score = ridge01.score(X_test, y_test)
    print("alpha=0.1-----------------------------------------------------")
    print("Training set score: {:.2f}".format(train_score))                     # 0.93
    print("Test set score: {:.2f}".format(test_score))                          # 0.77

    ridge = Ridge().fit(X_train, y_train)       # alpha=1.0 默认值
    train_score = ridge.score(X_train, y_train)
    test_score = ridge.score(X_test, y_test)
    print("alpha=1.0-----------------------------------------------------")
    print("Training set score: {:.2f}".format(train_score))                     # 0.89
    print("Test set score: {:.2f}".format(test_score))                          # 0.75

    ridge10 = Ridge(alpha=10.0).fit(X_train, y_train)
    train_score = ridge10.score(X_train, y_train)
    test_score = ridge10.score(X_test, y_test)
    print("alpha=10------------------------------------------------------")
    print("Training set score: {:.2f}".format(train_score))                     # 0.79
    print("Test set score: {:.2f}".format(test_score))                          # 0.64

    # 理解正则化对模型的影响
    # 方式一:alpha取值不同 岭回归 vs 线性模型
    plt.plot(ridge01.coef_, 'v', label="Ridge alpha=0.1")
    plt.plot(ridge.coef_, 's', label="Ridge alpha=1.0")
    plt.plot(ridge10.coef_, '^', label="Ridge alpha=10.0")
    plt.plot(lr.coef_, 'o', label="LinearRegression")
    plt.xlabel("Coefficient index")         # 系数索引
    plt.ylabel("Coefficient magnitude")     # 系数大小
    xlims = plt.xlim()                      # 获取x轴范围
    plt.hlines(0, xlims[0], xlims[1])    # 绘制y=0的基准线 x=[xlims[0],xlims[1]]
    plt.xlim(xlims)
    plt.ylim(-25, 25)
    plt.legend()
    plt.savefig("./../img/2.监督学习/12.不同alpha值的岭回归与线性回归的系数比较.png",dpi=1080)
    plt.show()

    """
    总结:
        x轴: x=0对应第一个特征的系数,x=1对应第二个特征的系数   
        y轴: 表示该系数的具体数值
        
        alpha=10:系数大多在-3和3之间
        alpha=1:系数值要稍大一点
        alpha=0.1:系数值的范围更大
        alpha=0:系数值的范围很大,许多点都超出了图像的范围------->线性回归
    """

    # 方式二:固定alpha值,但改变训练数据量
    mglearn.plots.plot_ridge_n_samples()
    plt.legend(loc=(0.15, 1.03), ncol=2, fontsize=9)
    plt.savefig("./../img/2.监督学习/13.岭回归和线性回归在波士顿房价数据集上的学习曲线.png",dpi=300)
    plt.show()

    """
    总结:
        对波士顿房价数据集做二次抽样,并在数据量逐渐增加的子数据集,使用此数据集对模型进行评估
        
        岭回归与线性回归的训练集分数都高于测试集
        岭回归的正则化导致其训练集分数要低于线性回归,但其测试集分数更高
        
        对于较小的数据集(数据量<400),线性回归学不到任何内容,但随着数据的增加,两个模型的性能都在提升,最终性能追上岭回归
        
        若训练集数据足够多,正则化则变得不那么重要,岭回归和线性回归具有相同性能
    """

    # 4.lasso boston 约束系数使其接近于0  若系数为0,则该特征被模型忽略掉--->特征选择
    lasso = Lasso().fit(X_train, y_train)       # alpha=1.0
    train_score = lasso.score(X_train, y_train)
    test_score = lasso.score(X_test, y_test)
    print("alpha=1.0------------------------------------------------------")
    print("Training set score: {:.2f}".format(train_score))                     # 0.29
    print("Test set score: {:.2f}".format(test_score))                          # 0.21
    print("Number of features used:", np.sum(lasso.coef_ != 0)) # L1正则化惩罚项  # 4         alpha=1,默认  欠拟合

    lasso001 = Lasso(alpha=0.01, max_iter=100000).fit(X_train, y_train)
    train_score = lasso001.score(X_train, y_train)
    test_score = lasso001.score(X_test, y_test)
    print("alpha=0.01------------------------------------------------------")
    print("Training set score: {:.2f}".format(train_score))                     # 0.90
    print("Test set score: {:.2f}".format(test_score))                          # 0.77
    print("Number of features used:", np.sum(lasso001.coef_ != 0))              # 33        alpha变小,性能变优

    lasso00001 = Lasso(alpha=0.0001, max_iter=100000).fit(X_train, y_train)
    train_score = lasso00001.score(X_train, y_train)
    test_score = lasso00001.score(X_test, y_test)
    print("alpha=0.0001------------------------------------------------------")
    print("Training set score: {:.2f}".format(train_score))                     # 0.95
    print("Test set score: {:.2f}".format(test_score))                          # 0.64                类似线性回归
    print("Number of features used:", np.sum(lasso00001.coef_ != 0))            # 96        alpha更小,消除正则化效果 过拟合

    plt.plot(lasso.coef_, 's', label="Lasso alpha=1")
    plt.plot(lasso001.coef_, '^', label="Lasso alpha=0.01")
    plt.plot(lasso00001.coef_, 'v', label="Lasso alpha=0.0001")
    plt.plot(ridge01.coef_, 'o', label="Ridge alpha=0.1")
    plt.legend(ncol=2, loc=(0, 1.05))
    plt.ylim(-25, 25)
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.savefig("./../img/2.监督学习/14.不同alpha值的lasso回归与岭回归的系数比较.png",dpi=1080)
    plt.show()

    """
    总结:
        alpha=1:大部分系数都是0,其他系数都很小
        alpha=0.01:大部分特征等于0 
        alpha=0.0001:正则化很弱的模型,大部分系数不为0,且很大
        
        alpha=0.1的岭回归预测性能与alpha=0.01的lasso性能类似,但岭回归的所有系数均不为0
        
    """

    """
    正则化:
        对模型做显式约束,避免过拟合
        (对系数w添加约束使得系数尽量小 w的所有元素都应接近于0    ------>   意味着每个特征对输出的影响应尽可能小)

    岭回归:(L2正则化) 
        系数w的平方和 

    lasso:(L1正则化)
        系数w的绝对值之和
        
    岭回归 vs lasso:
        实践中首先岭回归,若特征较多,你认为只有几个是重要的,则lasso效果更优
        ElasticNet 结合岭回归和lasso的惩罚项 L1正则化+L2正则化
    """