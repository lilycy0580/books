import torch
import torch as t
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


class MultiPerceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = Linear(in_features, hidden_features)  # 此处Linear是自定义Linear
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.nn.Module实现全连接层  y = Wx + b
    layer = Linear(4, 3)
    input = t.randn(2, 4)
    output = layer(input)
    print(output)
    """
    tensor([[ 2.9238,  3.0967,  0.4950],
            [-0.5025, -0.0759, -0.9008]], grad_fn=<AddBackward0>)
    """

    for name, param in layer.named_parameters():
        print(name, param)
    """
    w Parameter containing:
    tensor([[-1.1720, -0.3929,  0.5265],
            [ 1.1065,  0.9273, -1.7421],
            [-0.7699,  0.7864, -1.9963],
            [ 0.5836,  1.0392,  0.8023]], requires_grad=True)
    b Parameter containing:
    tensor([0.5269, 0.5730, 0.1390], requires_grad=True)
    """

    # 2.nn.Module实现多层感知机
    perceptron = MultiPerceptron(3,4,1)
    for name, param in perceptron.named_parameters():
        print(name, param)
    """
    layer1.w Parameter containing:
    tensor([[-0.3938,  1.5146,  1.4999,  0.1818],
            [ 0.2837,  1.2601,  1.1651, -0.8566],
            [-1.5661, -0.8520,  1.2770,  0.1278]], requires_grad=True)
    layer1.b Parameter containing:
    tensor([ 4.9097e-04, -7.1955e-01,  9.8246e-01, -1.0365e+00],
           requires_grad=True)
    layer2.w Parameter containing:
    tensor([[-1.1646],
            [ 0.0305],
            [-0.5842],
            [ 2.0983]], requires_grad=True)
    layer2.b Parameter containing:
    tensor([0.9039], requires_grad=True)    
    """
