
from chap1.config import np as cp
import numpy as np

if __name__ == '__main__':
    # 位精度
    a = np.random.randn(3)
    print(a.dtype)          # float64

    b = np.random.randn(3).astype(np.float32)
    print(b.dtype)          # float32
    c = np.random.randn(3).astype('f')
    print(c.dtype)          # float32

    # GPU (CuPy)
    x = np.arange(6).reshape(2, 3)
    print(x,x.shape)