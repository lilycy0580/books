import numpy as np
from chap3.common.layer import MatMul

if __name__ == '__main__':
    np.random.seed(42)                      # 设置随机种子,方便复现

    c = np.array([[1, 0, 0, 0, 0, 0, 0]])   # 输入    you 的向量化
    W = np.random.randn(7, 3)               # 权重
    h = np.dot(c, W)                        # 中间节点
    print(h)                                # [[ 0.49671415 -0.1382643   0.64768854]]   本质为提取权重W的对应行向量

    # 等价于
    c = np.array([[1, 0, 0, 0, 0, 0, 0]])
    W = np.random.randn(7, 3)
    layer = MatMul(W)
    h = layer.forward(c)
    print(h)                                # [[-0.2257763   0.0675282  -1.42474819]]

