

import timeit

import torch as t
import warnings

def for_loop_add(x,y):
    result = []
    for i,j in zip(x,y):            # Python的内置函数,接收两个(或更多)可迭代对象(如列表,元组),并将它们"打包"成一个元组的迭代器
        result.append(i+j)          # zip([1,2,3], [4,5,6])生成一个类似于[(1,4), (2,5), (3,6)]的迭代器
    return t.tensor(result)

# 1.向量化
if __name__ == '__main__':
    t.manual_seed(1000)
    warnings.filterwarnings('ignore')

    x = t.zeros(100)
    y = t.ones(100)
    time_for_loop = timeit.timeit(lambda: for_loop_add(x, y), number=100)
    time_vectorized = timeit.timeit(lambda: x + y, number=100)
    print(f"For循环加法执行100次耗时: {time_for_loop:.4f} 秒")
    print(f"向量化加法执行100次耗时: {time_vectorized:.4f} 秒")

    """
    For循环加法执行100次耗时: 0.0469 秒
    向量化加法执行100次耗时: 0.0002 秒
    """

