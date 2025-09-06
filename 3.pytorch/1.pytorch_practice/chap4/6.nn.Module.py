from collections import OrderedDict

import torch as t
import torch.nn as nn
from torchvision.models import VGG


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.param1 = nn.Parameter('param1',nn.Parameter(t.randn(3,3)))
        self.submodel1 = nn.Linear(3,4)

    def forward(self, input):
        x = self.param1.mm(input)
        x = self.submodel1(x)
        return x

def hook(module, input, output):
    features.copy_(output.data)  # 将该层的输出复制到features中

if __name__ == '__main__':
    t.manual_seed(1000)

    net = Net()
    print(net)

    print(net._modules)

    print(net._parameters)

    print(net.param1)

    for name,param in net.named_parameters():
        print(name,param.shape)

    for name,submodule in net._modules.items():
        print(name,submodule)

    bn = nn.BatchNorm1d(2)
    input = t.rand(3,2)
    output = bn(input)
    print(bn._buffers)

    input = t.arange(0,12).view(3,4).float()
    model = nn.Dropout()
    output = model(input)
    print(output)

    model.training = False
    output = model(input)
    print(output)

    print(net.training,net.submodel1.training)

    net.eval()
    print(net.training,net.submodel1.training)

    list = list(net.named_modules())
    print(list)

    # 钩子函数 一个预训练好的模型,需提取模型的某一层(非最后一层)的输出作为特性进行分类,不希望修改原有模型的定义文件
    model = VGG()
    features = t.Tensor()
    handle = model.layer8.register_forward_hook(hook)
    _ = model(input)
    handle.remove()

    module = nn.Module()
    module.param = nn.Parameter(t.ones(3,3))
    print(module._parameters)

    submodule1 = nn.Linear(3,3)
    submodule2 = nn.Linear(3,3)
    model_list = [submodule1,submodule2]
    module.submodels = model_list
    print('_modules:',module._models)
    print("__dict__['submodules']:",module.__dict__.get('submodules'))

    module_list = nn.ModuleList(model_list)
    module.submodels = module_list
    print('ModuleList is instance of nn.Module:',isinstance(module,nn.Module))
    print('_modules:',module._modules)
    print("__dict__['submodules']:",module.__dict__.get('submodules'))

    getattr(module,'training')  # 等价于module.training
    # print(module.__getattr__('training'))

    module.attr1 = 2
    getattr(module,'attr1')
    # print(module.__getattr__('attr1'))

    param = getattr(module,'param')
    print(param)

    # 7.模型的保存与加载
    # 方式一:
    t.save(net.state_dict(),'./net.pth')
    net2 = Net()
    net2.load_state_dict(t.load('./net.pth'))

    # 方式二:
    t.save(net2,'./net_all.pth')
    net2 = t.load('./net_all.pth')
    print(net2)

    # 8.GPU上运行模型
    new_net = nn.DataParallel(net,device_ids=[0,1])
    output = new_net(input)
    output = nn.parallel.data_parallel(new_net,input,device_ids=[0,1])






