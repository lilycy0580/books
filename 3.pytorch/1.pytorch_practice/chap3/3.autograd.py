import torch as t
from torch.autograd import Function
import numpy as np
from matplotlib import pyplot as plt


class MultiplyAdd(Function):
    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x)
        output = w * x + b
        return output

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors
        grad_w = grad_output * x
        grad_x = grad_output * w
        grad_b = grad_output * 1
        return grad_w, grad_x, grad_b


def f(x):
    y = x ** 2 * t.exp(x)
    return y


def gradf(x):
    dx = 2 * x * t.exp(x) + x ** 2 * t.exp(x)
    return dx


def abs(x):
    if x.data[0] > 0:
        return x
    else:
        return -x


def f(x):
    result = 1
    for i in x:
        if i.item() > 0:
            result *= i
    return result


def variable_hook(grad):
    print('y的梯度:', grad)


def get_fake_data(device, batch_size=8):
    x = t.rand(batch_size, 1) * 5
    x.to(device)
    y = x * 2  + 3 + t.randn(batch_size,1)
    return x, y


if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.autograd的用法 requires_grad和backward
    a = t.randn(3, 4, requires_grad=True)
    print(a.requires_grad)

    a = t.randn(3, 4)
    a.requires_grad = True
    print(a.requires_grad)

    b = t.ones(3, 4, requires_grad=True)
    c = (a + b).sum()  # 此处c依赖于a,而a需要求导,故c.requires_grad=True 自动设置
    c.backward()
    print(c)
    print(a.grad, b.grad)
    print(a.requires_grad, b.requires_grad, c.requires_grad)
    """
    True
    True
    tensor(12.8747, grad_fn=<SumBackward0>)
    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    True True True
    """

    print(a.is_leaf, b.is_leaf, c.is_leaf)

    a = t.rand(10, requires_grad=True)
    print(a.is_leaf)

    b = t.rand(10, requires_grad=True).cuda(0)
    print(b.is_leaf)

    c = t.rand(10, requires_grad=True) + 2
    print(c.is_leaf)

    d = t.rand(10).cuda(0)
    print(d.requires_grad, d.is_leaf)

    e = t.rand(10).cuda(0).requires_grad_()
    print(e.is_leaf)
    """
    True True False
    True
    False
    False
    False True
    True
    """

    # 查看autograd计算导数和手动推导的导数的区别: y = x^2 * exp(x)
    x = t.randn(3,4,requires_grad=True)
    y = f(x)
    print(y)
    y.backward(t.ones_like(y))
    assert t.all(x.grad == gradf(x))    # 若未抛出异常,则说明autograd与手动计算结果一直
    """
    tensor([[0.1106, 1.6212, 0.2470, 3.0834],
            [0.4788, 0.0522, 1.1152, 2.3008],
            [0.0320, 0.3228, 0.0610, 0.1101]], grad_fn=<MulBackward0>)
    """

    # 2.autograd原理 计算图
    x = t.ones(1)
    b = t.rand(1,requires_grad=True)    # 均匀分布U(0,1) 在[0,1]区间上连续均匀分布
    w = t.rand(1,requires_grad=True)
    y = w * x
    z = y + b
    print(x.requires_grad, b.requires_grad, w.requires_grad,y.requires_grad)
    """
    False True True True
    """

    print(z.grad_fn)                # grad_fn:查看Tensor的反向传播函数
    print(z.grad_fn.next_functions) # grad_fn.next_functions:grad_fn的输入,tuple类型
    print(y.grad_fn.next_functions)
    print(x.grad_fn,w.grad_fn)
    """
    <AddBackward0 object at 0x0000012CED67BBE0>
    ((<MulBackward0 object at 0x0000012CED67BF40>, 0), (<AccumulateGrad object at 0x0000012CED67BA60>, 0))
        y 乘法mul的输入,对应的反向传播函数y.grad_fn=MulBackward
        b 叶子节点,需要求导,对应的反向传播函数b.grad_fn=AccumulateGrad
    ((<AccumulateGrad object at 0x0000012CED67BA60>, 0), (None, 0))
        w 叶子节点,需要求导,梯度累加
        x 叶子节点,不需要求导,None
    None None
        叶子节点的grad_fn=None
    """

    z.backward(retain_graph=True)
    print(w.grad)
    z.backward()
    print(w.grad)
    """
    tensor([1.])
    tensor([2.])
    """

    x = t.ones(1,requires_grad=True)
    y = abs(x)
    y.backward()
    print(x.grad)
    """
    tensor([1.])
    """

    x = t.arange(-2, 4).float().requires_grad_()
    y = f(x)
    y.backward()
    print(x.grad)
    """
    tensor([0., 0., 0., 6., 3., 2.])
    """

    x = t.ones(1,requires_grad=True)
    w = t.rand(1,requires_grad=True)
    y = x * w
    print(x.requires_grad,w.requires_grad,y.requires_grad)

    # 2种方式关闭自动求导: y虽依赖w和x,但是y.requires_grad=False
    with t.no_grad():
        x = t.ones(1)
        w = t.rand(1,requires_grad=True)
        y = x * w
        print(x.requires_grad,w.requires_grad,y.requires_grad)
    """
    True True True
    False True False
    """

    t.set_grad_enabled(False)
    x = t.ones(1)
    w = t.rand(1,requires_grad=True)
    y = x * w
    print(x.requires_grad,w.requires_grad,y.requires_grad)
    """
    False True False
    """

    t.set_grad_enabled(True)

    a = t.ones(3,4,requires_grad=True)
    b = t.ones(3,4,requires_grad=True)
    c = a * b
    print(a.data,a.data.requires_grad)
    """
    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])
    False                           a.data.requires_grad 独立于计算图
    """

    x = t.ones(3,requires_grad=True)
    w = t.rand(3,requires_grad=True)
    y = x * w
    z = y.sum()
    print(x.requires_grad,w.requires_grad,y.requires_grad)
    z.backward()
    print(x.grad,w.grad,y.grad)
    """
    True True True
    tensor([0.3189, 0.6136, 0.4418]) tensor([1., 1., 1.]) None  非叶子节点的梯度不会被保存,y.grad=None
    """

    # 方式一:使用grad获取中间变量的梯度
    x = t.ones(3,requires_grad=True)
    w = t.rand(3,requires_grad=True)
    y = x * w
    z = y.sum()
    grad = t.autograd.grad(z,y)
    print(grad)
    """
    (tensor([1., 1., 1.]),)
    """

    # 方式二:使用hook
    x = t.ones(3,requires_grad=True)
    w = t.rand(3,requires_grad=True)
    y = x * w
    hook_handle = y.register_hook(variable_hook)    # 注册hook
    z = y.sum()
    z.backward()
    hook_handle.remove()    # 除非每次都使用hook,否则用完之后请移除hook
    """
    y的梯度: tensor([1., 1., 1.])
    """

    # 3.autograd扩展:Function
    x = t.ones(1)
    w = t.rand(1,requires_grad=True)
    b = t.rand(1,requires_grad=True)
    z = MultiplyAdd.apply(w, x, b)      # 前向传播
    z.backward()                              # 反向传播
    print(x.grad,w.grad,b.grad)
    """
    None tensor([1.]) tensor([1.])      x不需要求导,但是中间过程会计算它的导数,但随后被清空
    """


    x = t.ones(1)
    w = t.rand(1,requires_grad=True)
    b = t.rand(1,requires_grad=True)
    z = MultiplyAdd.apply(w, x, b)      # 前向传播
    print(z.grad_fn)                          # 计算图的梯度函数,查看如何进行反向传播
    print(z.grad_fn.apply(t.ones(1)))         # 反向传播,调用MultiplyAdd.backward(),输出grad_w, grad_x, grad_b
    """
    <torch.autograd.function.MultiplyAddBackward object at 0x000002703182D440>
    (tensor([1.]), tensor([0.4418], grad_fn=<MulBackward0>), tensor([1.]))
    """

    # 4.利用autograd实现线性回归
    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')

    x, y = get_fake_data(device)
    plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
    plt.show()
    plt.savefig('fake_data_autograd.png')

    w = t.rand(1,1,requires_grad=True)
    b = t.zeros(1,requires_grad=True)
    losses = np.zeros(500)
    lr = 0.005
    for i in range(500):
        x,y = get_fake_data(device)

        # 前向传播
        y_pred = x.mm(w) + b.expand_as(y)
        loss = 0.5*(y_pred - y)**2
        loss = loss.sum()
        losses[i] = loss.item()

        # 反向传播
        loss.backward()

        # 更新参数
        w.data.sub_(lr * w.grad.data)
        b.data.sub_(lr * b.grad.data)

        # 梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

        if i % 50 == 0:
            x = t.arange(0,6).float().view(-1,1)
            y = x.mm(w.data) + b.data.expand_as(x)
            plt.plot(x.cpu().numpy(), y.cpu().numpy())  # 预测结果

            x2, y2 = get_fake_data(device,batch_size=200)
            plt.scatter(x2.cpu().numpy(), y2.cpu().numpy())   # 真实数据

            plt.xlim(0,5)
            plt.ylim(0,13)
            plt.show()
            plt.pause(0.5)
            plt.savefig('linear_regression_autograd.png')
    print(f'w:{w.item():.3f}, b:{b.item():.3f}')

    plt.plot(losses)
    plt.ylim(5,50)
    plt.show()
    plt.savefig('linear_regression_autograd_loss.png')
