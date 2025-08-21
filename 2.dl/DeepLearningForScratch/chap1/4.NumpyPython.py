from queue import PriorityQueue

import numpy as np

if __name__ == '__main__':
    x = np.array([1, 2, 3])
    print(x)
    print(type(x))                              # <class 'numpy.ndarray'>

    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    print(x+y)
    print(x-y)
    print(x*y)                                  # [1. 4. 9.]
    print(x/y)                                  # [1. 1. 1.]
    print(x/2)                                  # [0.5 1.  1.5]

    A = np.array([[1, 2], [3, 4]])
    print(A)
    print(A.shape)
    print(A.dtype)                              # int32

    B = np.array([[3, 0], [0, 6]])
    print(A+B)
    print(A*B)                                  # 对应元素相加/相乘

    print(A)
    print(A*10)

    # np.array() 生成N维数组 一维数组为向量,二维数组为矩阵,三维及以上为张量/多维数组

    # 广播
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[10, 20]])
    print(A*B)

    # 获取元素
    X = np.array([[1, 2], [3, 4]])
    print(X)
    print(X[0])
    print(X[0][1])

    for row in X:
        print(row)

    X = X.flatten()
    print(X)                                    # [1 2 3 4]
    Y = X[np.array([0,2])]
    print(Y)                                    # [1 3]
    print(X>1)                                  # [False  True  True  True]
    print(X[X>3])                               # [4]




