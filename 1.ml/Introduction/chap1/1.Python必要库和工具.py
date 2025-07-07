
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import pandas as pd
import mglearn

if __name__ == '__main__':
    # Python必要库和工具
    # numpy
    x = np.array([[1, 2, 3], [4, 5, 6]])
    print("x:\n",x)

    # scipy
    # 对角线元素为1,其余元素为0======> 单位矩阵/对角矩阵
    eye = np.eye(4)
    print("NumPy array:\n", eye)

    # 稀疏矩阵 CSR与COO格式
    sparse_matrix = sparse.csr_matrix(eye)
    print("SciPy sparse CSR matrix:\n", sparse_matrix)

    data = np.ones(4)           # [1. 1. 1. 1.]
    row_indices = np.arange(4)  # [0 1 2 3]
    col_indices = np.arange(4)  # [0 1 2 3]
    eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
    print("COO representation:\n", eye_coo)

    data = np.array([1,2,3,6])
    row_indices = np.array([1,1,4,5])
    col_indices = np.array([2,3,6,3])
    eye_coo = sparse.coo_matrix((data, (row_indices, col_indices)))
    print("COO representation:\n", eye_coo)

    # matplotlib
    x = np.linspace(-10, 10, 100)   # [-10,10]区间共生成100个数
    y = np.sin(x)
    plt.plot(x, y, marker="x")
    plt.savefig("./../img/1.引言/1.用matplotlib画出正弦函数的简单折线图.png",dpi=1080)
    plt.show()

    # pands
    data = {
                'Name': ["John", "Anna", "Peter", "Linda"],
                'Location': ["New York", "Paris", "Berlin", "London"],
                'Age': [24, 13, 53, 33]
            }
    data_pandas = pd.DataFrame(data)
    print(data_pandas)
    print(data_pandas[data_pandas.Age > 30])


