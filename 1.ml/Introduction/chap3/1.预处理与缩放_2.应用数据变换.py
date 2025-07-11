from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    # 2.应用数据变换
    # KSVM应用在cancer上,使用MinMaxScaler进行预处理
    cancer = load_breast_cancer()                              # (426, 30)  (143, 30)
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target,random_state=1)
    # 训练集需要先fit在transform,测试集直接transform
    scaler = MinMaxScaler().fit(X_train)        # 据X_train中的数据计算后续转换所需的参数
    X_train_scaled = scaler.transform(X_train)  # 使用前面学习到的参数实际转换训练数据
    X_test_scaled = scaler.transform(X_test)
    print("transformed shape: {}".format(X_train_scaled.shape))
    print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
    print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
    print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
    print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))
    print("test per-feature minimum after scaling:\n{}".format(X_test_scaled.min(axis=0)))
    print("test per-feature maximum after scaling:\n{}".format(X_test_scaled.max(axis=0)))
    # 总结:训练集缩放后所有特征在[0,1],测试集缩放后min<0,max>1 因为使用的是训练集的min与range进行缩放

