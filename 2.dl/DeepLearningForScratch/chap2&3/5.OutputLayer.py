import numpy as np


def softmax(a):
    c = np.max(a)
    exp = np.exp(a-c)       # 防溢出
    sum_exp = np.sum(exp)
    y = exp/sum_exp
    return y

if __name__ == '__main__':
    # 可能会有溢出
    a = np.array([0.3, 2.9, 4.0])
    exp_a = np.exp(a)           # [ 1.34985881 18.17414537 54.59815003]
    sum_exp_a = np.sum(exp_a)   # 74.1221542101633
    y = exp_a / sum_exp_a       # [0.01821127 0.24519181 0.73659691]
    print(exp_a, sum_exp_a, y)

    # 防溢出
    a = np.array([0.3, 2.9, 4.0])
    c = np.max(a)
    exp_a = np.exp(a-c)         # [0.02472353 0.33287108 1.        ]
    sum_exp_a = np.sum(exp_a)   # 1.3575946101684189
    y = exp_a / sum_exp_a       # [0.01821127 0.24519181 0.73659691]
    print(exp_a, sum_exp_a,y)

    a = np.array([0.3, 2.9, 4.0])
    y = softmax(a)              # [0.01821127 0.24519181 0.73659691]
    print(y,np.sum(y))          # 1.0



    


