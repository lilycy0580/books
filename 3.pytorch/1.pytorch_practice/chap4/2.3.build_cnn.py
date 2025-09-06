import torch as t
import torch.nn as nn
from collections import OrderedDict

# ModuleList是nn.Module的子类,当在主module中使用它时,ModuleList能够被主module识别为子module
# list中的子module不能被主module识别,ModuleList中的子module可以被主module识别
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.list = [nn.Linear(3,4), nn.ReLU()]
        self.module_list = nn.ModuleList([nn.Conv2d(3, 3,3),
                                          nn.ReLU()])

    def forward(self, x):
        pass


if __name__ == '__main__':
    t.manual_seed(1000)

    # 2.3.1 Sequential的使用
    net1 = nn.Sequential()
    net1.add_module('conv', nn.Conv2d(3, 3, 3))
    net1.add_module('batchnorm', nn.BatchNorm2d(3))
    net1.add_module('activation_layer', nn.ReLU())

    net2 = nn.Sequential(
        nn.Conv2d(3, 3, 3),
        nn.BatchNorm2d(3),
        nn.ReLU()
    )

    net3 = nn.Sequential(
        OrderedDict([
            ('conv1', nn.Conv2d(3, 3, 3)),
            ('bn1', nn.BatchNorm2d(3)),
            ('relu1', nn.ReLU())
        ])
    )

    print(net1, net2, net3)
    """
    Sequential(
      (conv): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (batchnorm): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (activation_layer): ReLU()
    ) 
    Sequential(
      (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    ) 
    Sequential(
      (conv1): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
      (bn1): BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu1): ReLU()
    )
    """

    # 获取子module
    print(net1.conv, net2[0], net3.conv1)
    """
    Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)) 
    Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1)) 
    Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
    """

    input = t.rand(1, 3, 4, 4)
    output1 = net1(input)
    output2 = net2(input)
    output3 = net3(input)
    output4 = net3.relu1(net1.batchnorm(net1.conv(input)))
    print(output1.shape, output2.shape, output3.shape, output4.shape,
          output1 == output2, output2 == output3, output3 == output4)
    """
    torch.Size([1, 3, 2, 2]) 
    torch.Size([1, 3, 2, 2]) 
    torch.Size([1, 3, 2, 2]) 
    torch.Size([1, 3, 2, 2]) 
    
    tensor([[[[False, False],
              [False, False]],
             [[ True, False],
              [False,  True]],
             [[False, False],
              [ True,  True]]]]) 
              
    tensor([[[[False, False],
              [False, False]],
             [[ True, False],
              [False, False]],
             [[False, False],
              [ True,  True]]]]) 
              
    tensor([[[[ True,  True],
              [False, False]],
             [[ True, False],
              [False, False]],
             [[False, False],
              [ True,  True]]]])
    """

    # 2.3.2 ModuleList的使用
    modellist = nn.ModuleList([
        nn.Linear(3, 4),
        nn.ReLU(),
        nn.Linear(4, 2)])
    input = t.rand(1, 3)
    for model in modellist:
        input = model(input)
    # output = modellist(input)
    # print(output)
    """
    NotImplementedError: Module [ModuleList] is missing the required "forward" function
    """

    model = MyModel()
    print(model)
    for name,param in model.named_parameters():
        print(name, param.size())
    """
    MyModel(
      (module_list): ModuleList(
        (0): Conv2d(3, 3, kernel_size=(3, 3), stride=(1, 1))
        (1): ReLU()
      )
    )
    module_list.0.weight torch.Size([3, 3, 3, 3])
    module_list.0.bias torch.Size([3])
    """