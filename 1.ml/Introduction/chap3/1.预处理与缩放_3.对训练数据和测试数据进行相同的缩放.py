import mglearn
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # 3.对训练数据和测试数据进行相同的缩放
    # 原始数据集
    X, _ = make_blobs(n_samples=50, centers=5, random_state=4, cluster_std=2)   # 5分类
    X_train, X_test = train_test_split(X, random_state=5, test_size=0.1) # 90%,10%
    # 使用训练集对测试集进行缩放 (使用训练集对训练集进行缩放)
    train_scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = train_scaler.transform(X_train)
    X_test_scaled = train_scaler.transform(X_test)
    # 使用测试集对测试集进行缩放 (使用训练集对训练集进行缩放)
    test_scaler = MinMaxScaler().fit(X_test)
    X_test_scaled_useTest = test_scaler.transform(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
    axes[0].scatter(X_test[:, 0], X_test[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
    axes[0].legend(loc='upper left')
    axes[0].set_title("Original Data")
    axes[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="Training set", s=60)
    axes[1].scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], marker='^', c=mglearn.cm2(1), label="Test set", s=60)
    axes[1].set_title("Scaled Data use Train")
    axes[2].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=mglearn.cm2(0), label="training set", s=60)
    axes[2].scatter(X_test_scaled_useTest[:, 0], X_test_scaled_useTest[:, 1], marker='^', c=mglearn.cm2(1), label="test set", s=60)
    axes[2].set_title("Scaled Data use Test ")
    for ax in axes:
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
    fig.tight_layout()
    plt.savefig("./../img/3.无监督学习与预处理/2.对左图中的训练数据和测试数据同时缩放的效果(中)和分别缩放的效果(右).png",dpi=1080)
    plt.show()

    """
    总结:
        第一张:原始数据集 
        第二张:MinMaxScaler缩放,fit作用在训练集上,transform作用在训练集和测试集上 
              与第一张图看起来完成相同,仅坐标轴刻度发生变换 
              所有特征都位于0到1之间,测试点最值并非0,1
        第三张:分别对训练集和测试集进行缩放,发现所有特征最值均为0,1
              测试集相对训练集的移动不一致,随意改变数据点的排列,应避免
    """

