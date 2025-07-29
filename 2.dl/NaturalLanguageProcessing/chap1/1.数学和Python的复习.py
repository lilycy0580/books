import numpy as np

if __name__ == '__main__':

    # 1.向量,矩阵,张量
    x = np.array([1, 2, 3])
    print(x.__class__)      # <class 'numpy.ndarray'>
    print(x.shape)          # (3,)
    print(x.ndim)           # 1
    print(x.dtype)          # int64

    W = np.array([[1, 2, 3], [4, 5, 6]])
    print(W.__class__)
    print(W.shape)
    print(W.ndim)
    print(W.dtype)

    W = np.array([[1, 2, 3], [4, 5, 6]])
    X = np.array([[0, 1, 2], [3, 4, 5]])
    SUM = W + X
    PRODUCT = W * X
    print(SUM)
    print(PRODUCT)
    """
        [[ 1  3  5]
         [ 7  9 11]]
         
        [[ 0  2  6]
         [12 20 30]]     
    """

    # 2.广播
    A = np.array([[1, 2], [3, 4]])
    b = np.array([10, 20])
    print(A*10)
    print(A*b)
    """
        [[10 20]
         [30 40]]
         
        [[10 40]
         [30 80]]    
    """

    # 3.向量内积和矩阵内积
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    print(np.dot(a, b))             # 32
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(np.dot(A, B))
    """
        [[19 22]
         [43 50]]    
    """

    # 4.矩阵检查
    """
    A       B       =       C
    3×2     2×4             3×4
    """



