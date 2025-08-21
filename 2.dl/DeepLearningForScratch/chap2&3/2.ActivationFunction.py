import numpy as np
from matplotlib import pyplot as plt


# sigmod
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# # 实数
# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

# Numpy数组
def step_function(x):
    y = x>0
    return y.astype(int)

def relu(x):
    return np.maximum(0, x)

if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    y = sigmoid(x)
    print(y)

    x = np.array([-1.0, 1.0, 2.0])
    y = step_function(x)
    print(y)

    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(x)
    y2 = step_function(x)
    y3 = relu(x)
    plt.figure(figsize=(10, 6))
    plt.plot(x,y1,label='sigmoid')
    plt.plot(x,y2,label='step_function',linestyle='--')
    plt.plot(x,y3,label='relu',linestyle='-.')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()







