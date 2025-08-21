import numpy as np

# 与门(没有权重与偏置)
def AND_(x1, x2):
    w1,w2,theta = 0.5,0.5,0.7
    tmp = w1*x1 + w2*x2
    if tmp <= theta:
        return 0
    else:
        return 1

# 与门(使用权重与偏置)
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b= 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 异或门
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y

if __name__ == '__main__':

    # Numpy实现感知机
    x = np.array([0,1])
    w = np.array([0.5,0.5])
    b = -0.7
    print(w*x)
    y = np.sum(w*x)+b
    print(y)

    print(XOR(0,0))     # 0
    print(XOR(0, 1))    # 1
    print(XOR(1,0))     # 1
    print(XOR(1,1))     # 0



