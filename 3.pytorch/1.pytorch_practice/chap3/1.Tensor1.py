
import torch as t
import numpy as np
"""
    1.Tensor的基本操作
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.创建Tensor
    a = t.Tensor(2,3)
    b = t.Tensor([[1,2,3],[4,5,6]])
    list_b = b.tolist()
    c = t.Tensor(t.randn(2,3))
    print(a,'\n',b,'\n',c,'\n',list_b)
    """
    tensor([[0., 0., 0.],
        [0., 0., 0.]])
    tensor([[1., 2., 3.],
        [4., 5., 6.]])
    tensor([[-1.1720, -0.3929,  0.5265],
        [ 1.1065,  0.9273, -1.7421]])
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    """

    b_size = b.size()
    b1 = t.Tensor(b_size)
    b2 = t.Tensor((2,3))
    print(b1,'\n',b2)
    """
    tensor([[0., 0., 0.],
        [0., 0., 0.]])
    tensor([2., 3.])
    """

    a1 = t.Tensor()
    a2 = t.tensor([])
    print(a1,'\n',a2)
    """
    tensor([])
    tensor([])
    """

    a1 = t.Tensor([2, 3])
    a2 = t.tensor([2,3])
    print(a1.type(),'\n',a2.type())
    """
    torch.FloatTensor
    torch.LongTensor
    """

    arr = np.ones((2,3),dtype=np.float64)
    a = t.tensor(arr)
    print(a)
    """
    tensor([[1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    """

    a = t.ones(2,3)
    b = t.zeros(2,3)
    c = t.eye(2,3,dtype=t.int32)
    print(a,'\n',b,'\n',c)
    """
    tensor([[1., 1., 1.],
        [1., 1., 1.]])
    tensor([[0., 0., 0.],
        [0., 0., 0.]])
    tensor([[1, 0, 0],
        [0, 1, 0]], dtype=torch.int32)
    """
    a = t.tensor([[1,2,3],[4,5,6]])
    b = t.ones_like(a)
    print(a,'\n',b)
    """
    tensor([[1, 2, 3],
        [4, 5, 6]])
    tensor([[1, 1, 1],
        [1, 1, 1]])
    """

    a = t.arange(1,6,2)
    b = t.linspace(1,10,3)
    print(a,'\n',b)
    """
    tensor([1, 3, 5])
    tensor([ 1.0000,  5.5000, 10.0000])
    """

    a = t.randn(2,3)
    b = t.randperm(5)
    print(a,'\n',b)
    """
    tensor([[-0.7699,  0.7864, -1.9963],
        [ 0.5836,  1.0392,  0.8023]])
    tensor([2, 3, 1, 4, 0])
    """

    a = t.tensor((),dtype=t.int32,device=t.device('cpu'))
    a.new_ones((2,3))
    print(a)
    """
    tensor([], dtype=torch.int32)
    """

    sum1 = a.numel()
    sum2 = a.nelement()
    print(sum1,'\n',sum2)
    """
    0
    0
    """

    # 2.Tensor的类型
    # 更改tensor类型 FloatTensor <---> DoubleTensor
    a = t.rand(2,3)
    print(a.dtype)
    t.set_default_tensor_type('torch.DoubleTensor')
    a = t.rand(2,3)
    print(a.dtype)
    t.set_default_tensor_type('torch.FloatTensor')
    """
    torch.float32
    torch.float64
    """

    b1 = a.type(t.FloatTensor)
    b2 = a.float()
    b3 = a.type_as(b1)
    print(a.dtype,'\n',b1.dtype,'\n',b2.dtype,'\n',b3.dtype)
    """
    torch.float64
    torch.float32
    torch.float32
    torch.float32
    """

    # new_*() 使用的构造函数为DoubleTensor
    a.new_ones(2,4)
    print(a.dtype)
    """
    torch.float64
    """

    # new_*() 复制Tensor的device
    a = t.randn(2,3).cuda()
    b = a.new_ones(2,4)
    print(b)
    """
    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.]], device='cuda:0')
    """

    # 3.索引操作
    a = t.randn(3, 4)
    print(a)
    print('第1行结果:',a[0])
    print('第2列结果:',a[:,1])
    print('查看第2行最后两个元素:',a[1,-2:])
    """
    tensor([[-1.1720, -0.3929,  0.5265,  1.1065],
            [ 0.9273, -1.7421, -0.7699,  0.7864],
            [-1.9963,  0.5836,  1.0392,  0.8023]])
    第1行结果: tensor([-1.1720, -0.3929,  0.5265,  1.1065])
    第2列结果: tensor([-0.3929, -1.7421,  0.5836])
    查看第2行最后两个元素: tensor([-0.7699,  0.7864])
    """

    print(a>0)
    print((a>0).int())
    """
    tensor([[False, False,  True,  True],
            [ True, False, False,  True],
            [False,  True,  True,  True]])
    tensor([[0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 1]], dtype=torch.int32)
    """

    print(a[a>0])
    print(a.masked_select(a>0))
    print(t.where(a>0,a,t.zeros_like(a)))
    """
    tensor([0.5265, 1.1065, 0.9273, 0.7864, 0.5836, 1.0392, 0.8023])
    tensor([0.5265, 1.1065, 0.9273, 0.7864, 0.5836, 1.0392, 0.8023])
    tensor([[0.0000, 0.0000, 0.5265, 1.1065],
            [0.9273, 0.0000, 0.0000, 0.7864],
            [0.0000, 0.5836, 1.0392, 0.8023]])
    """

    # gather:按照index获取Tensor元素 eg:获取对角线与反对角线元素
    a = t.arange(0,16).view(4,4)
    print(a)
    """
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11],
            [12, 13, 14, 15]])
    """

    index = t.tensor([[0,1,2,3]])
    b = a.gather(0,index)
    print(b)
    """
    tensor([[ 0,  5, 10, 15]])
    """

    index = t.tensor([[0,1,2,3]]).t()
    b = a.gather(1,index)
    print(b)
    """
    tensor([[ 0],
            [ 5],
            [10],
            [15]])
    """

    index = t.tensor([[0,1,2,3],[3,2,1,0]]).t()
    b = a.gather(1,index)
    print(index,b)
    """
    tensor([[0, 3],
            [1, 2],
            [2, 1],
            [3, 0]])
    tensor([[ 0,  3],
            [ 5,  6],
            [10,  9],
            [15, 12]])
    """

    # scatter_:按照index将数据写入Tensor,gather逆操作
    c =  t.zeros(4,4).long()
    c.scatter_(1, index, b)
    print(c)
    """
    tensor([[ 0,  0,  0,  3],
            [ 0,  5,  6,  0],
            [ 0,  9, 10,  0],
            [12,  0,  0, 15]])
    """

    # item()
    item = t.Tensor([1.]).item()
    print(item)
    # item = t.Tensor([1,2]).item()   会报错,因为item()仅对包含一个元素的Tensor适用
    """
    1.0
    """

    # 4.拼接操作
    a = t.arange(6).view(2,3)
    b1 = t.cat((a,a),0)
    b2 =t.cat((a,a),1)
    print(b1,'\n',b2)
    """
    tensor([[0, 1, 2],
            [3, 4, 5],
            [0, 1, 2],
            [3, 4, 5]])
    tensor([[0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5]])
    """

    b = t.stack((a,a),0)
    print(a.shape,b.shape,b)
    """
    torch.Size([2, 3])
    torch.Size([2, 2, 3])
    tensor([[[0, 1, 2],
             [3, 4, 5]],
            [[0, 1, 2],
             [3, 4, 5]]])
    """

    # 5.高级索引 操作结果与原Tensor不共享内存
    a = t.arange(0,16).view(2,2,4)
    print(a)
    print(a[[1,0],[1,1],[2,0]])
    print(a[[1,0],[0],[1]])
    """
    tensor([[[ 0,  1,  2,  3],
             [ 4,  5,  6,  7]],
            [[ 8,  9, 10, 11],
             [12, 13, 14, 15]]])
    tensor([14,  4])                a[1, 1, 2] 与 a[0, 1, 0]
    tensor([9, 1])                  a[1, 0, 1] 与 a[0, 0, 1]
    """

    # 6.逐元素操作   clamp(x,min,max) 截断
    a = t.arange(0,6).float().view(2,3)
    cosa = t.cos(a)
    print(cosa)
    """
    tensor([[ 1.0000,  0.5403, -0.4161],
            [-0.9900, -0.6536,  0.2837]])
    """

    print(a%3)
    print(t.fmod(a,3))
    """
    tensor([[0., 1., 2.],
            [0., 1., 2.]])
    tensor([[0., 1., 2.],
            [0., 1., 2.]])
    """

    print(a)
    print(t.clamp(a,min=2,max=4))
    """
    tensor([[0., 1., 2.],
            [3., 4., 5.]])
    tensor([[2., 2., 2.],
            [3., 4., 4.]])
    """

    # 7.归并操作
    a = t.ones(2,3)
    sum1 = a.sum(dim=0, keepdim=True)
    print(sum1,sum1.shape)
    sum2 = a.sum(dim=0, keepdim=False)
    print(sum2,sum2.shape)
    """
    tensor([[2., 2., 2.]]) torch.Size([1, 3])
    tensor([2., 2., 2.]) torch.Size([3])
    """

    a = t.arange(2,8).view(2,3)
    print(a)
    print(a.cumsum(dim=1))
    """
    tensor([[2, 3, 4],
            [5, 6, 7]])
    tensor([[ 2,  5,  9],
            [ 5, 11, 18]])
    """

    # 8.比较操作
    a = t.linspace(0,15,6).view(2,3)
    b = t.linspace(15,0,6).view(2,3)
    print(a>b)
    print('a中大于b的元素:',a[a>b])
    print('a中最大的元素:',t.max(a))
    """
    tensor([[False, False, False],
            [ True,  True,  True]])
    a中大于b的元素: tensor([ 9., 12., 15.])
    a中最大的元素: tensor(15.)
    """

    max1 = t.max(b,dim=1)
    max2 = t.max(a,b)
    print(max1,'\n',max2)
    """
    torch.return_types.max(
        values=tensor([15.,  6.]),
        indices=tensor([0, 0]))
    tensor([[15., 12.,  9.],
            [ 9., 12., 15.]])
    """

    a = t.tensor([1,2,3,4,5])
    b = t.topk(a,3)
    print(b)
    """
    torch.return_types.topk(
        values=tensor([5, 4, 3]),
        indices=tensor([4, 3, 2]))
    """

    a = t.randn(2,3)
    b = t.argsort(a,dim=1)
    print(a,'\n',b)
    """
    tensor([[-1.1720, -0.3929,  0.5265],
            [ 1.1065,  0.9273, -1.7421]])
    tensor([[0, 1, 2],
            [2, 1, 0]])
    """

    # 比较两个整型Tensor可用==直接比较,有精度限制的浮点数,使用t.allclose()进行比较
    a = t.tensor([1.000001,1.000001,0.999999])
    b = t.ones_like(a)
    print(a==b)
    print(t.allclose(a,b))
    """
    tensor([False, False, False])
    True
    """

