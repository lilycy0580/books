import torch
import torch as t
# 数据预处理
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
# 定义网络,损失函数和优化器
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == '__main__':
    t.manual_seed(1000)
    """
    cifar-10图像分类任务:
        1.使用torchvision加载并预处理CIFAR-10数据集
        2.定义网络
        3.定义损失函数和优化器
        4.训练网络并更新网络参数
        5.测试网络
    """
    # 1.加载并预处理数据集
    transforms = transforms.Compose([transforms.ToTensor(),                                             # 转换为Tensor
                                     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])  # 归一化

    trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms)
    trainloader = t.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms)
    testloader = t.utils.data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    """
    图片展示,tensor与图像数据格式互换
    """
    classes = ('plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck')

    (data,label) = trainset[100]
    print(classes[label])               # ship

    show = ToPILImage()                 # 转换器类,将张量/numpy转为PIL图像
    if data.min() < 0:                  # 此时数据归一化到[-1,1],将其重新缩放到[0,1],方便显示
        image = show((data + 1) / 2)
    else:
        image = show(data)
    image = image.resize((100, 100))
    image.show()

    dataiter = iter(trainloader)        # 将trainloader转为迭代器,方便逐个获取批次
    images,labels = next(dataiter)      # 获取一批次的数据  images:[batch_size, channels, height, width]
    print(' '.join('%11s' % classes[labels[j]] for j in range(4)))
    imgs = (images + 1) / 2             #                   torch.Size([4, 3, 32, 32])      4张图片 3×32×32
    imgs = tv.utils.make_grid(imgs)     # 将图片排成网格状     torch.Size([3, 36, 138])        1张图片 3×36×138
    imgs = show(imgs)                   # 将tensor转为PIL图像
    imgs = imgs.resize((100, 100))      #                   (100, 100)                      1张图片 3*100*100
    imgs.show()

    tensors = transforms(imgs)
    print(tensors.shape)
    """
           frog         cat        deer         car
    torch.Size([3, 100, 100])
    """

    # 2.定义网络
    net = Net()
    print(net)
    """
    Net(
      (conv1): Conv2d(3, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )    
    """

    # 3.定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 4.训练网络    输入数据 ———> 前向传播,反向传播 ———> 更新参数
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print('[%d,  %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 1000))
                running_loss = 0.0
    print('Finished Training')
    """
    [1,   2000] loss: 4.358
    [1,   4000] loss: 3.798
    [1,   6000] loss: 3.428
    [1,   8000] loss: 3.204
    [1,  10000] loss: 3.067
    [1,  12000] loss: 2.970
    [2,   2000] loss: 2.820
    [2,   4000] loss: 2.742
    [2,   6000] loss: 2.702
    [2,   8000] loss: 2.626
    [2,  10000] loss: 2.599
    [2,  12000] loss: 2.582
    Finished Training
    """

    # gpu
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    net.to(device)
    images, labels = next(iter(testloader))
    images, labels = images.to(device), labels.to(device)
    outputs = net(images)
    loss = criterion(outputs, labels)
    print(loss)
    """
    tensor(0.5801, device='cuda:0', grad_fn=<NllLossBackward0>)
    """

    # 5.测试模型
    dataiter = iter(testloader)
    images,labels = next(dataiter)
    print('实际的label:',' '.join('%08s'%classes[labels[j]] for j in range(4)))
    show(tv.utils.make_grid(images / 2 - 0.5)).resize((400, 100)).show()

    net.to("cpu")
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    print('预测的label:',' '.join('%08s'%classes[predicted[j]] for j in range(4)))
    """
    实际的label:    horse    truck     ship     frog
    预测的label:    horse    truck      car     frog
    """

    correct = 0
    total = 0
    with t.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct +=(predicted==labels).sum()
    print('1000张图像的准确率为:%f %%' %(100*correct//total))
    """
    1000张图像的准确率为:52.000000 %
    """