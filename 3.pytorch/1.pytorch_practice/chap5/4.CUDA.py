
import torch as t
import torch.nn as nn

class VeryBigModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.GiantParameter1 = t.nn.Parameter(t.randn(100000, 20000)).to('cuda:0')  # param1在cuda0 两个参数所占空间较大,需放两块卡上
        self.GiantParameter2 = t.nn.Parameter(t.randn(20000, 100000)).to('cuda:1')  # param2在cuda1

    def forward(self, x):
        x = self.GiantParameter1.mm(x.cuda(0))
        x = self.GiantParameter2.mm(x.cuda(1))
        return x

"""
4.CUDA
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    tensor = t.Tensor(3, 4)
    tensor.cuda(0)              # 返回一个新的Tensor,保存在第1块GPU上,原来的Tensor并没有改变
    print(tensor.is_cuda)
    """
    False
    """

    tensor = tensor.cuda()      # 不指定所使用的GPU设备,默认使用第1块GPU
    print(tensor.is_cuda)
    """
    True
    """

    module = nn.Linear(3, 4)
    module.cuda(device=0)
    print(module.weight.is_cuda)
    """
    True
    """

    tensor = t.Tensor(3, 4).to('cuda:0')
    print(tensor.is_cuda)
    """
    True
    """

    criterion = t.nn.CrossEntropyLoss(weight=t.Tensor([1, 3]))
    input = t.randn(4, 2).cuda()
    target = t.Tensor([1, 0, 0, 1]).long().cuda()
    # loss = criterion(input, target)       # 这行会报错,因weight未被转移至GPU
    criterion.cuda()                        # 不会报错
    loss = criterion(input, target)
    print(criterion._buffers)
    """
    {'weight': tensor([1., 3.], device='cuda:0')}
    """

    x = t.cuda.FloatTensor(2, 3)            # 如果未指定使用哪块GPU,则默认使用GPU 0   x.get_device() == 0
    y = t.FloatTensor(2, 3).cuda()          #                                    y.get_device() == 0

    with t.cuda.device(0):                  # 指定默认使用GPU 0
        a = t.cuda.FloatTensor(2, 3)        # 在GPU 0上构建Tensor
        b = t.FloatTensor(2, 3).cuda()      # 将Tensor转移至GPU 0
        assert a.get_device() == b.get_device() == 0

        c = a + b
        assert c.get_device() == 0

        z = x + y
        assert z.get_device() == 0

        d = t.randn(2, 3).cuda(0)           # 手动指定使用GPU 0
        assert d.get_device() == 0

    t.set_default_tensor_type('torch.cuda.FloatTensor')  # 指定默认Tensor的类型为GPU上的FloatTensor
    a = t.ones(2, 3)
    print(a.is_cuda)
    """
    True
    """

    # 指定Tensor加载的设备
    device1 = t.device('cpu')
    device2 = t.device('cuda:0')
    device3 = t.device("cuda" if t.cuda.is_available() else "cpu")  # 推荐
    print(device1,device2,device3)

    device = t.device("cuda" if t.cuda.is_available() else "cpu")  # 推荐
    x = t.empty((2, 3)).to(device)
    print(x.device)
    """
    cpu 
    cuda:0 
    cuda
    cuda:0    
    """

    x_cpu = t.empty(2, device='cpu')
    x_gpu = t.empty(2, device=device)
    print(x_cpu, x_cpu.is_cuda)
    print(x_gpu, x_gpu.is_cuda)
    """
    tensor([1.3027e-26, 1.2486e-42], device='cpu') False
    tensor([1., 3.]) True
    """

    y_cpu = x_cpu.new_full((3, 4), 3.1415)
    y_gpu = x_gpu.new_zeros(3, 4)
    print(y_cpu, y_cpu.is_cuda)
    print(y_gpu, y_gpu.is_cuda)
    """
    tensor([[3.1415, 3.1415, 3.1415, 3.1415],
            [3.1415, 3.1415, 3.1415, 3.1415],
            [3.1415, 3.1415, 3.1415, 3.1415]], device='cpu') False
    tensor([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]]) True
    """

    z_cpu = t.ones_like(x_cpu)
    z_gpu = t.zeros_like(x_gpu)
    print(z_cpu, z_cpu.is_cuda)
    print(z_gpu, z_gpu.is_cuda)
    """
    tensor([1., 1.], device='cpu') False
    tensor([0., 0.]) True
    """



