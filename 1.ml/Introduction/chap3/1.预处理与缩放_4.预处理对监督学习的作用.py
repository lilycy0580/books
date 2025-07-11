from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC

if __name__ == '__main__':
    # 4.预处理对监督学习的作用
    # KSVC cancer 原始分布/MinMaxScaler/StandardScaler
    # 原始分布
    cancer = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
    svm = SVC(C=100)
    svm.fit(X_train, y_train)
    train_score = svm.score(X_train, y_train)
    print("Test set accuracy: {:.2f}".format(train_score))      # 0.94
    # MinMaxScaler
    scaler = MinMaxScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm.fit(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    print("Scaled test set accuracy: {:.2f}".format(test_score)) # 0.97
    # StandardScaler
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    svm.fit(X_train_scaled, y_train)
    test_score = svm.score(X_test_scaled, y_test)
    print("Scaled test set accuracy: {:.2f}".format(test_score)) # 0.96  数据缩放后提高模型精度



