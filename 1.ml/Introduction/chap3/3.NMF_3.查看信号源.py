import mglearn
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import NMF, PCA
from scipy import signal
if __name__ == '__main__':
    # 3.查看信号源
    # 无法观测到原始信号,只能观测到三个信号的叠加混合 S
    # S = mglearn.datasets.make_signals()
    random_state = np.random.RandomState(42)
    n_samples = 2000
    time = np.linspace(0, 8, n_samples)
    # 创建3种信号
    s1 = np.sin(2 * time)                   # 正弦波信号
    s2 = np.sign(np.sin(3 * time))          # 方波信号
    s3 = signal.sawtooth(2 * np.pi * time)  # 锯齿波信号
    # 三个信号的叠加混合并添加噪声
    S = np.c_[s1, s2, s3]                   # 将s1,s2,s3拼接成一个矩阵
    S += 0.2 * random_state.normal(size=S.shape) # 矩阵S的每个元素添加高斯(正态分布)随机噪声,噪声的幅度为0.2
    S /= S.std(axis=0)                      # 标准化数据
    S -= S.min()                            # 归一化

    plt.figure(figsize=(6, 1))
    plt.plot(S, '-')
    plt.xlabel("Time")
    plt.ylabel("Signal")
    plt.savefig("./../img/3.无监督学习与预处理/18.原始信号源.png", dpi=1080)
    plt.show()

    # 将混合信号分解为原始分量
    # 预处理:将数据混合成100维的状态
    A = np.random.RandomState(0).uniform(size=(100, 3)) # 均匀分布 (100, 3)
    X = np.dot(S, A.T)                                  # (2000, 3)*(3, 100)
    print("Shape of measurements: {}".format(X.shape))  # (2000, 100)

    # 方式一:NMF还原三信号
    nmf = NMF(n_components=3, random_state=42)
    S_ = nmf.fit_transform(X)
    print("Recovered signal shape: {}".format(S_.shape)) # (2000, 3)  NMF降维

    # 方式二:PCA还原三信号
    pca = PCA(n_components=3)
    H = pca.fit_transform(X)                             # (2000, 3)  PCA降维

    # 绘制NMF与PCA
    models = [S,  X,  S_, H]
    names = ['True sources','Observations (first three measurements)', 'NMF recovered signals','PCA recovered signals']
    fig, axes = plt.subplots(4, figsize=(8, 4), gridspec_kw={'hspace': .5}, subplot_kw={'xticks': (), 'yticks': ()})
    for model, name, ax in zip(models, names, axes):
        ax.set_title(name)
        ax.plot(model[:, :3], '-')
    plt.savefig("./../img/3.无监督学习与预处理/19.利用NMF和PCA还原混合信号源.png", dpi=1080)
    plt.show()

    """
    总结:
        数据来源来自X的100次测量中的3次
        NMF在发现原始信号源时得到了不错的结果,而PCA则失败,仅使用第一个成分来解释数据中的大部分变化
        NMF生成的分量是没有顺序的,在这个例子中,NMF分量的顺序与原始信号完全相同(参见三条曲线的颜色),但这纯属偶然
        
        还有许多其他算法可用于将每个数据点分解为一系列固定分量的加权求和,如独立成分分析(ICA),因子分析(FA)和稀疏编码(字典学习)  超出本书范围
    """