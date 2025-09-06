
import torch as t
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    t.manual_seed(1000)

    # 使用nn.init初始化
    linear = nn.Linear(3, 4)
    w = init.xavier_normal_(linear.weight)
    print(w)
    """
    Parameter containing:
    tensor([[-1.0670,  0.3120,  0.5555],
            [ 0.4288,  0.2816,  0.3063],
            [ 0.0743, -0.6287,  0.0455],
            [ 0.0836,  0.9566, -0.1101]], requires_grad=True)
    """

    # 使用公式直接初始化
    std = 2 ** 0.5/7**0.5
    w = linear.weight.data.uniform_(0, std)
    print(w)
    """
    tensor([[0.2241, 0.0018, 0.2133],
            [0.5234, 0.5063, 0.3325],
            [0.4874, 0.0189, 0.3424],
            [0.2004, 0.5276, 0.4183]])    
    """

    net = Net()
    for name, param in net.named_parameters():
        if name.find('linear') != -1:
            param[0]
            param[1]
        elif name.find('conv') != -1:
            pass
        elif name.find('norm') != -1:
            pass
