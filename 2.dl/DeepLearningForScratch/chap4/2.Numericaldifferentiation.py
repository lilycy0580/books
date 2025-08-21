import numpy as np
from matplotlib import pyplot as plt


# 数值微分(不好的案例)
def numerical_diff_bad(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h
"""
改进点:
    改进点1 h=10e-50,引入舍入误差  10e-4即可
    改进点2 计算函数f在x+h和x之间的差分 真导数计算f在x处的斜率,该处导数对应(x+h)与x之间的斜率
           数值微分有误差,为减少误差计算f在(x-h)和(x+h)之间的差分 中心差分
           (x + h)和x之间的差分为前向差分
"""

def numerical_diff(f, x):
    h = 1e-4                        # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

def f1(x):
    return 0.01*x**2 + 0.1*x

def f2(x):
    return x[0]**2 + x[1]**2

# 求f2在x[0]=3,x[1]=4的偏导数
def f2_x0(x0):
    return x0**2 + 4.0**2

def f2_x1(x1):
    return x1**2 + 3.0**2

if __name__ == '__main__':

    # 数值微分与偏导数  f(x) = 0.01*x**2 + 0.1*x  计算f(x)在x=5和x=10处的导数
    print(numerical_diff(f1,5))     # 数值微分 0.1999999999990898    导数  0.2
    print(numerical_diff(f1,10))    # 数值微分 0.2999999999986347    导数  0.3

    # 偏导数  f(x0,x1) = x0**2 + x1**2
    print(numerical_diff(f2_x0,3.0))    # 6.00000000000378
    print(numerical_diff(f2_x1,4.0))    # 7.999999999999119

    x = np.arange(0.0, 20.0, 0.1)  # 以0.1为单位，从0到20的数组x
    y = f1(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()
    plt.savefig('./2.png')



