
import torch as t
"""
基本索引
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.元组序列
    a = t.arange(1,25).reshape(2, 3,4)
    print(a)
    """
    tensor([[[ 1,  2,  3,  4],
             [ 5,  6,  7,  8],
             [ 9, 10, 11, 12]],

            [[13, 14, 15, 16],
             [17, 18, 19, 20],
             [21, 22, 23, 24]]])
    """

    index_value = a[0,1,2]      # 等价于a[(0, 1, 2)]
    print(index_value)
    """
    tensor(7)
    """

    index_value = a[1, 1]
    print(index_value)
    """
    tensor([17, 18, 19, 20])
    """

    value1 = a[0, 1, 2]
    value2 = a[[0, 1, 2]]
    value3 = a[(0, 1, 2),]
    print(value1)
    print(value2)
    print(value3)
    """
    tensor(7)

     Traceback (most recent call last):
      File "D:\books\3.pytorch\1.pytorch_practice\chap6\3.1_basic_index.py", line 35, in <module>
        value2 = a[[0, 1, 2]]
    IndexError: index 2 is out of bounds for dimension 0 with size 2

    Traceback (most recent call last):
     File "D:\books\3.pytorch\1.pytorch_practice\chap6\3.1_basic_index.py", line 36, in <module>
        value3 = a[(0, 1, 2),]
    IndexError: index 2 is out of bounds for dimension 0 with size 2
    """

    # 2.:和...
    a = t.randn(64,3,224,224)
    print(a[:, :, 0:224:4, :].shape)
    print(a[:, :, ::4, :].shape)
    print(a[..., ::4, :].shape)
    """
    torch.Size([64, 3, 56, 224])
    torch.Size([64, 3, 56, 224])
    torch.Size([64, 3, 56, 224])
    """

    # 3.None索引
    a = t.randn(2,3,4,5)
    print(a.unsqueeze(0).shape)
    print(a[None,...].shape)
    """
    torch.Size([1, 2, 3, 4, 5])
    torch.Size([1, 2, 3, 4, 5])
    """

    b = a.unsqueeze(1)          # torch.Size([2, 1, 3, 4, 5])
    b = b.unsqueeze(3)          # torch.Size([2, 1, 3, 1, 4, 5])
    b = b.unsqueeze(5)          # torch.Size([2, 1, 3, 1, 4, 1, 5])
    print(b.shape)
    c = a[:,None,:,None,:,None,:]
    print(c.shape)
    """
    torch.Size([2, 1, 3, 1, 4, 1, 5])
    torch.Size([2, 1, 3, 1, 4, 1, 5])
    """

    a = t.arange(16*256).view(16,256)       # torch.Size([16, 256])
    a = a.unsqueeze(1)                      # torch.Size([1, 16, 256])
    a_T = a.transpose(2,1)      # torch.Size([1, 256, 16])
    a_matrix = a_T @ a                      # @表示矩阵的乘法
    print(a_matrix.shape)
    """
    torch.Size([16, 256, 256])
    """

    a = t.arange(16 * 256).view(16, 256)
    b = a[:,:,None] * a[:,None,:]           # 16*256*1  16*1*256  广播法则
    c = a[:,None,:] * a[:,:,None]           # 16*1*256  16*256*1  广播法则
    print(b.shape,c.shape)
    """
    torch.Size([16, 256, 256]) 
    torch.Size([16, 256, 256])
    """

    assert t.equal(b,c)
    assert t.equal(a_matrix,b)






