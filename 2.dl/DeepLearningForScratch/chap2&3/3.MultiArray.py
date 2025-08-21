import numpy as np

if __name__ == '__main__':
    # 一维数组 矩阵 张量
    A = np.array([1, 2, 3, 4])
    print(A)
    print(A.shape,np.ndim(A))

    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(B)
    print(B.shape,np.ndim(B))

    # 矩阵乘法
    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([[1, 2, 3, 4],[ 5 ,6, 7, 8]])
    C = np.dot(A,B)
    print(C,C.shape)    # (3, 4)

    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([7, 8])
    C = np.dot(A,B)
    print(C,C.shape)    # [23 53 83] (3,)

    # 神经网络内积
    X = np.array([[1, 2]])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    Y = np.dot(X,W)
    print(Y,Y.shape)    # [[ 5 11 17]] (1, 3)







