import torch as t
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import math
import warnings

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


class MyLinear(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.weight = nn.Parameter(t.randn(3, 4))
        self.bias = nn.Parameter(t.zeros(3))

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    t.manual_seed(1000)

    # nn.Module vs nn.functional
    input = t.randn(2, 3)
    model = nn.Linear(3, 4)
    output1 = model(input)
    output2 = F.linear(input, model.weight, model.bias)
    bool = output1.equal(output2)
    print(bool)
    """
    True
    """

    b1 = F.relu(input)
    b2 = nn.ReLU()(input)
    bool = b1.equal(b2)
    print(bool)
    """
    True
    """

    # 采样函数 torch.nn.functional.grid_sample 对Tensor进行双线性采样,将输出变为用户想要的形状
    to_pil = ToPILImage()
    to_tensor = ToTensor()

    # lena原始图像
    lena = Image.open('./lena.png')
    lena = to_tensor(lena)  # torch.Size([1, 200, 200])
    to_pil(lena.data.squeeze(0)).show()

    # lena旋转90°
    lena = lena.unsqueeze(0)
    angle = -90 * math.pi / 180
    theta = t.tensor([[math.cos(angle), math.sin(-angle), 0],
                     [math.sin(angle), math.cos(angle), 0]],dtype=t.float)
    grid = F.affine_grid(theta.unsqueeze(0), lena.size())
    out = F.grid_sample(lena, grid=grid, mode='bilinear')
    to_pil(out.data.squeeze(0)).show()

