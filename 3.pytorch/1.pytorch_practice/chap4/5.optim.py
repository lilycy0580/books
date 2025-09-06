import torch as t
import torch.nn as nn
import torch.optim as optim


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.w = nn.Parameter(t.randn(in_features, out_features))
        self.b = nn.Parameter(t.randn(out_features))

    def forward(self, x):
        x = x.mm(self.w)
        return x + self.b.expand_as(x)


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.layer1 = Linear(in_features, hidden_features)
        self.layer2 = Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.layer1(x)
        x = t.sigmoid(x)
        x = self.layer2(x)
        return x


if __name__ == '__main__':
    # 方式一: 为网络设置学习率
    net = Perceptron(3, 4, 1)
    optimizer = optim.SGD(net.parameters(), lr=1)
    optimizer.zero_grad()  # 梯度清零

    input = t.randn(32, 3)
    output = net(input)
    output.backward(output)  # 真正的反向传播在下一步执行
    optimizer.step()  # 执行优化,更新参数

    # 方式二: 为不同的参数分别设置不同的学习率
    weight_param = [param for name, param in net.named_parameters() if name.endwith('.W')]
    bias_param = [param for name, param in net.named_parameters() if name.endwith('.b')]
    optimizer = optim.SGD([
        {'params': weight_param, 'lr': 0.01},
        {'params': bias_param, 'lr': 0.01}
    ], lr=1e-5, )

    # 调整学习率,构建一个新的optimizer
    prev_lr = 0.1
    optimizer1 = optim.SGD([
        {'params': bias_param},
        {'params': weight_param, 'lr': prev_lr * 0.1}
    ], lr=1e-5)

    # 手动衰减学习率,保存动量
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

