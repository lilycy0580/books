import torch as t
from einops import rearrange, reduce
"""
爱因斯坦操作
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.einsum  爱因斯坦求和
    a = t.arange(9).view(3, 3)
    b = t.einsum('ij->ji', a)  # 直接交换两个维度
    print(a)
    print(b)
    """
    tensor([[0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]])
    tensor([[0, 3, 6],
            [1, 4, 7],
            [2, 5, 8]])
    """

    a = t.arange(36).view(3, 4, 3)
    b = t.einsum('ijk->', a) # 所有元素求和
    print(a)
    print(b)
    """
    tensor([[[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]],
            [[12, 13, 14],[15, 16, 17],[18, 19, 20],[21, 22, 23]],
            [[24, 25, 26],[27, 28, 29],[30, 31, 32],[33, 34, 35]]])
    tensor(630)
    """

    a = t.arange(36).view(3, 4, 3)              # a的下标 ijk
    b = t.arange(24).view(4, 3, 2)              # b的下标 jim
    c = t.einsum('ijk,jim->km', a, b)     # 对下标i和j进行求和,得到k*m的输出
    print(a)
    print(b)
    print(c)
    """
    tensor([[[ 0,  1,  2],[ 3,  4,  5],[ 6,  7,  8],[ 9, 10, 11]],
            [[12, 13, 14],[15, 16, 17],[18, 19, 20],[21, 22, 23]],
            [[24, 25, 26],[27, 28, 29],[30, 31, 32],[33, 34, 35]]])
    tensor([[[ 0,  1],[ 2,  3],[ 4,  5]],
            [[ 6,  7],[ 8,  9],[10, 11]],
            [[12, 13],[14, 15],[16, 17]],
            [[18, 19],[20, 21],[22, 23]]])
    tensor([[2640, 2838],
            [2772, 2982],
            [2904, 3126]])
    """

    a = t.arange(6).view(2, 3)
    b = t.arange(3)
    sum = t.einsum('ij,j->ij', a, b)    # 矩阵对应维度相乘,b进行了广播
    print(a)
    print(b)
    print(sum)
    """
    tensor([[0, 1, 2],
            [3, 4, 5]])
    tensor([0, 1, 2])
    tensor([[ 0,  1,  4],
            [ 0,  4, 10]])
    """

    a = t.arange(6).view(2, 3)
    b = t.arange(6).view(3, 2)
    c_in = t.einsum('ij,ij->', a, a)    # 内积,结果是一个数
    c_out = t.einsum('ik,kj->ij', a, b) # 外积,矩阵乘法的结果
    print(c_in)
    print(c_out)
    """
    tensor(55)
	tensor([[10, 13],
        	[28, 40]])
    """

    # 2.einops
