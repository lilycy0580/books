import torch as t
import numpy as np

if __name__ == '__main__':
    # 1.Tensor常见操作
    t.manual_seed(1000)

    print(t.__version__)        # 2.5.0

    x1 = t.Tensor(2, 3)         # 仅分配空间未初始化
    x2 = t.tensor([1, 2, 3])    # 确切数值进行初始化
    print(x1,x1.size())
    print(x2,x2.size())
    """
    tensor([[-8.1964e+03,  1.6942e-42,  0.0000e+00],
            [ 0.0000e+00,  0.0000e+00,  0.0000e+00]]) torch.Size([2, 3])
    tensor([1, 2, 3]) torch.Size([3])        
    """

    x = t.rand(2,3)             # 正态分布
    print(x,x.shape,x.size()[1])
    """
    tensor([[0.3189, 0.6136, 0.4418],
            [0.2580, 0.2724, 0.6261]]) torch.Size([2, 3]) 3
    """

    # 加法
    x = t.rand(2, 3)
    y = t.rand(2,3)
    z1 = x+y
    print(z1)

    z2 = t.add(x,y)
    print(z2)

    z3 = t.Tensor(2,3)
    t.add(x,y,out=z3)
    print(z3)
    """
    tensor([[0.8627, 0.4906, 1.0353],
            [0.6543, 1.0390, 0.5280]])
    tensor([[0.8627, 0.4906, 1.0353],
            [0.6543, 1.0390, 0.5280]])
    tensor([[0.8627, 0.4906, 1.0353],
            [0.6543, 1.0390, 0.5280]])
    """

    # inplace操作,修改Tensor本身
    y.add(x)                # 返回一个新的Tensor
    print(y)
    y.add_(x)               # 修改Tensor本身
    print(y)
    """
    tensor([[0.4217, 0.1254, 0.6818],
            [0.0571, 0.6818, 0.0473]])
    tensor([[0.8627, 0.4906, 1.0353],
            [0.6543, 1.0390, 0.5280]])    
    """

    # 索引操作
    print(x[:,1])
    """
    tensor([0.3653, 0.3572])
    """

    # Tensor和NumPy数组互换,大多数情况下共享内存,一改全改
    a = t.tensor(2)
    print(a)

    b = a.numpy()
    print(b)

    c = t.from_numpy(b)
    print(c)

    c.add_(1)
    print(b)
    print(c)
    """
    tensor(2)
    2
    tensor(2)
    3
    tensor(3)    
    """

    # 获取Tensor中的某个元素的值,通过索引操作获取一个零维度的Tensor,即scalar
    a = np.ones(5)
    b = t.from_numpy(a)
    scalar = b[0]
    print(scalar,scalar.shape,scalar.size(),scalar.item())

    tensor = t.tensor([1])
    print(tensor, tensor.shape, tensor.size(), tensor.item())
    """
    tensor(1., dtype=torch.float64) torch.Size([]) torch.Size([]) 1.0
    tensor([1]) torch.Size([1]) torch.Size([1]) 1
    """

    # 内存共享       torch.from_numpy() 与 tensor.detach()
    # 内存不共享      torch.tensor() 与 tensor.clone()
    tensor = t.tensor([3,4])
    old_tensor = tensor
    new_tensor = old_tensor.clone()
    new_tensor[0]=1111
    print(old_tensor,new_tensor)

    new_tensor = old_tensor.detach()
    new_tensor[0] = 1111
    print(old_tensor, new_tensor)
    """
    tensor([3, 4]) tensor([1111,    4])
    tensor([1111,    4]) tensor([1111,    4])   
    """

    # Tensor维度变换
    #   维度变换:
    #       view        仅适用于内存中连续存储的Tensor, tensor.contiguous.view()
    #       reshape     推荐
    #   维度交换:
    #       permute     枚举所有维度
    #       transpose   枚举交换维度
    x = t.randn(4,4)
    y1 = x.view(16)
    y2 = x.view(-1,8)
    y3 = x.reshape(-1,8)
    print(y1.size(),y2.size(),y3.size(),y1.shape,y2.shape,y3.shape)
    """
    torch.Size([16]) torch.Size([2, 8]) torch.Size([2, 8]) torch.Size([16]) torch.Size([2, 8]) torch.Size([2, 8])
    """

    x = t.randn(2,4,6)
    change1 = t.permute(x,(0,2,1))
    change2 = t.transpose(x,0,2)
    change3 = x.permute((0, 2, 1))
    change4 = x.transpose(0, 2)
    print(f'x size {change1.size()}')
    print(f'x size {change2.size()}')
    print(f'x size {change3.size()}')
    print(f'x size {change4.size()}')
    """
    x size torch.Size([2, 6, 4])
    x size torch.Size([6, 4, 2])
    x size torch.Size([2, 6, 4])
    x size torch.Size([6, 4, 2])
    """

    # 操作Tensor的维度
    #   tensor.squeeze()    维度压缩
    #   tensor.unsqueeze()  维度扩展
    #   torch.cat()         维度拼接
    x = t.randn(3,2,4,1)
    y = x.squeeze(-1)
    z = x.unsqueeze(0)
    o = t.cat([x,x],0)
    print(x.shape,y.shape,z.shape,o.shape)
    """
    torch.Size([3, 2, 4, 1]) torch.Size([3, 2, 4]) torch.Size([1, 3, 2, 4, 1]) torch.Size([6, 2, 4, 1])
    """

    # CPU与GPU互换 使用GPU时,将数据加载到显存上
    x = t.randn(3,2)
    y = t.randn(3,2)
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    x = x.to(device)
    y = y.to(device)
    z = x + y
    print(z)
    """
    tensor([[ 0.9741,  0.7299],
            [ 1.5725, -1.0014],
            [-0.2582,  0.1449]], device='cuda:0')
    """


























