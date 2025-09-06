
import torch as t
import torch.nn as nn

if __name__ == '__main__':
    t.manual_seed(1000)

    # 2.2 激活函数
    relu = nn.ReLU(inplace=True)    # 如果一个Tensor只作为激活层的输入使用,则该激活层可设为inplace=True
    input = t.randn(2,3)
    print(input)
    """
    tensor([[-1.1720, -0.3929,  0.5265],
            [ 1.1065,  0.9273, -1.7421]])
    """

    output = relu(input)
    print(output)
    """
     tensor([[0.0000, 0.0000, 0.5265],
            [1.1065, 0.9273, 0.0000]])   
    """