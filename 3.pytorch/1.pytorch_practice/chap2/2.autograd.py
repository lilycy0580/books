import torch as t

if __name__ == '__main__':
    # 2.autograd自动微分
    t.manual_seed(1000)

    # autograd模块:自动反向传播
    # optim模块:各种优化方法,常见的梯度下降优化方法

    x = t.ones(4, 2, requires_grad=True)
    y = x.sum()
    print(y, y.grad_fn)

    y.backward()
    print(x.grad)

    y.backward()
    y.backward()
    y.backward()
    print(x.grad)

    x.grad.data.zero_()
    print(x.grad)

    y.backward()
    print(x.grad)
    """
    tensor(8., grad_fn=<SumBackward0>) <SumBackward0 object at 0x000002554F6AFD90>
    tensor([[1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.]])
    tensor([[4., 4.],
            [4., 4.],
            [4., 4.],
            [4., 4.]])    
    tensor([[0., 0.],
            [0., 0.],
            [0., 0.],
            [0., 0.]])
    tensor([[1., 1.],
            [1., 1.],
            [1., 1.],
            [1., 1.]])
    """

    a = t.randn(2, 2)
    a = (a * 3) / (a - 1)
    print(a.requires_grad)

    a.requires_grad = True
    print(a.requires_grad)

    b = (a * a).sum()
    b.backward()
    print(a.grad,b.grad_fn)
    """
    False
    True
    tensor([[ 3.2376,  1.6924],
            [-6.6710, 62.3328]]) <SumBackward0 object at 0x0000024202F7FD90>
    """