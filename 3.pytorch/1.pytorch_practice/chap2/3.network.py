import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0],-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    # 3.神经网络
    t.manual_seed(1000)

    # torch.nn 神经网络的模块化接口
    #   nn.Module 神经网络的封装,包含神经网络各层的定义以及前向传播方法

    """
    LetNet网络: 7层
        输入:32×32的图像
        输出:10维
        组成:2个卷积神经网络,2个下采样,3个全连接
        训练网络:
            1.定义一个包含可学习参数的神经网络
            2.加载数据集
            3.进行前向传播,获取网络的输出结果,计算损失(网络输出结果与正确结果之间的差距)
            4.进行反向传播,更新网络参数
            5.保存网络模型
    """

    # 1.定义网络
    """
    定义网络时,模型需继承nn.Module,并实现forward() 与 __init__()
        某层包含可学习的参数需放入构造函数 __init__()
        某层不含有学习的参数放入前向传播 forward(),并用nn.functional实现
        
    torch.nn仅支持mini-batch,若只输入一个样本,则使用input.unsqueeze(0)将batch_size设为1
        nn.Conv2d输入必须是4维度,nSamples × nChannels × Height × Width,若只输入一个样本,则将nSamples设为1
    """

    net = Net()
    print(net)
    """
    Net(
      (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
      (conv2): Conv2d(6, 16, kernel_size=(5, 5), stride=(1, 1))
      (fc1): Linear(in_features=400, out_features=120, bias=True)
      (fc2): Linear(in_features=120, out_features=84, bias=True)
      (fc3): Linear(in_features=84, out_features=10, bias=True)
    )
    """

    params = list(net.parameters())
    print(len(params))
    """
    10
    """

    for name, param in net.named_parameters():
        print(name,":", param.shape)
    """
    conv1.weight : torch.Size([6, 1, 5, 5])
    conv1.bias : torch.Size([6])
    conv2.weight : torch.Size([16, 6, 5, 5])
    conv2.bias : torch.Size([16])
    fc1.weight : torch.Size([120, 400])
    fc1.bias : torch.Size([120])
    fc2.weight : torch.Size([84, 120])
    fc2.bias : torch.Size([84])
    fc3.weight : torch.Size([10, 84])
    fc3.bias : torch.Size([10])
    """

    input = t.randn(1,1,32,32)
    out = net(input)
    print(out.shape)
    """
    torch.Size([1, 10])
    """

    net.zero_grad()                 # 梯度清零
    out.backward(t.ones(1,10))      # 反向传播

    # 2.损失函数
    """
    torch.nn实现了大多数损失函数:
        nn.MSELoss()            均方误差
        nn.CrossEntropyLoss()   交叉熵损失
    
    LetNet的loss计算图:
        input --> conv2d --> relu --> maxpool2d --> conv2d --> relu --> maxpool2d
              --> view --> linear --> relu --> linear --> relu --> linear
              --> MSELoss 
              --> loss 
    """
    output = net(input)
    target = t.arange(0,10).view(1,10).float()
    criterion = nn.MSELoss()
    loss = criterion(output, target)
    print(loss)

    net.zero_grad()
    print("反向传播前covn1.bias的梯度:",net.conv1.bias.grad)
    print("反向传播前covn1.bias的数值:", net.conv1.bias.data)
    loss.backward()
    print("反向传播后covn1.bias的梯度:",net.conv1.bias.grad)
    print("反向传播前covn1.bias的数值:", net.conv1.bias.data)
    """
    tensor(28.5525, grad_fn=<MseLossBackward0>)
    反向传播前covn1.bias的梯度: None
    反向传播前covn1.bias的数值: tensor([ 0.0410,  0.1132, -0.0106, -0.0735,  0.1544,  0.1035])
    反向传播后covn1.bias的梯度: tensor([ 0.0360, -0.0363,  0.0305,  0.0239, -0.0368, -0.0133])
    反向传播前covn1.bias的数值: tensor([ 0.0410,  0.1132, -0.0106, -0.0735,  0.1544,  0.1035])
    """

    # 3.优化器
    """
    完成反向传播后,需使用优化方法更新网络的权重和参数 
    torch.optim模块:
        RMSprop 
        Adam 
        SGD  weight = weight - learning_rate * gradient
        
    更新:
      loss.backward()  仅更新梯度
      optimizer.step() 仅更新参数    
    """
    # SGD手动实现
    learning_rate = 0.01
    for param in net.parameters():
        param.data.sub_(learning_rate * param.grad.data)


    optimizer = optim.SGD(net.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    output = net(input)
    loss = criterion(output, target)
    print("反向传播前covn1.bias的梯度:",net.conv1.bias.grad)
    loss.backward()
    print("反向传播后covn1.bias的梯度:",net.conv1.bias.grad)
    print("参数更新前covn1.bias的数值:", net.conv1.bias.data)
    optimizer.step()
    print("参数更新后covn1.bias的数值:", net.conv1.bias.data)
    """
    反向传播前covn1.bias的梯度: None
    反向传播后covn1.bias的梯度: tensor([-0.0055, -0.1209,  0.0189, -0.0240, -0.1588, -0.0898])
    参数更新前covn1.bias的数值: tensor([ 0.0406,  0.1136, -0.0109, -0.0737,  0.1548,  0.1037])
    参数更新后covn1.bias的数值: tensor([ 0.0407,  0.1148, -0.0111, -0.0735,  0.1564,  0.1046])
    """

    # 4.数据加载与处理
    """
    数据加载与预处理:
        Dataset
        DataLoader 
    视觉工具包:
        torchvision
            datasets        常用数据集    
            models          网络结构与预训练模型
            transforms      数据预处理 Tensor与PIL Image
    """

