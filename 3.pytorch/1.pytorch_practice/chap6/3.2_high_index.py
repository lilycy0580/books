import torch as t
import torch.nn as nn
from PIL import Image
import numpy as np

def Conv_base(img, filters, stride, padding):
    '''
    img: 输入图像 channel×height×width
    filters: 卷积核 input_channel×output_channel×height×width
    stride: 卷积核的步长
    padding: 边缘填充的大小
    '''
    Cin, Hin, Win = img.shape
    _, Cout, K, _ = filters.shape

    # 计算卷积输出的大小
    Hout = ((Hin + 2 * padding - K) / stride).long() + 1
    Wout = ((Win + 2 * padding - K) / stride).long() + 1

    # 首先构建一个输出的样子
    col = t.zeros(Cin, K, K, Hout, Wout)
    # 通过padding的值将imgs进行扩充
    imgs = nn.ZeroPad2d(padding.item())(img)
    for h in range(Hout):
        for w in range(Wout):
            h1 = int(h * stride.item())
            w1 = int(w * stride.item())
            col[..., h, w] = imgs[:, h1:h1 + K, w1:w1 + K]
    col = col.view(Cin * K * K, Hout * Wout)
    # 将卷积核变形
    filters = filters.transpose(1, 0).view(Cout, Cin * K * K)
    out_img = (filters @ col).view(Cout, Hout, Wout)
    return out_img

def Conv(img, filter, stride=1, padding=0):
    '''
    img: 形状为 channel_in×height×width
    filter:形状为 channel_in×channel_out×kernel×kernel
    '''
    Cin, Hin, Win = img.shape
    Cout, K = filter.shape[1], filter.shape[2]
    # 计算卷积输出图像的参数,默认stride=1,padding=0
    Hout = ((t.tensor(Hin) + 2 * padding - K) / stride).long() + 1
    Wout = ((t.tensor(Win) + 2 * padding - K) / stride).long() + 1

    # 卷积核下标的索引
    K1 = t.arange(-(K // 2), K // 2 + 1)
    idx11, idx12 = t.meshgrid(K1, K1)
    # 输出Tensor下标索引
    H = t.linspace(K // 2, K // 2 + stride * (Hout - 1), Hout).long()
    W = t.linspace(K // 2, K // 2 + stride * (Wout - 1), Wout).long()
    idx21, idx22 = t.meshgrid(H, W)
    # 两种索引的组合形式
    idx1 = idx11[:, :, None, None] + idx21[None, None, :, :]
    idx2 = idx12[:, :, None, None] + idx22[None, None, :, :]

    # 改变filter的形状,便于接下来的矩阵相乘
    filter = filter.transpose(0, 1).reshape(Cout, Cin * K * K)
    # 输入图像经过整数数组索引后改变成适合矩阵乘法的形状
    img = img[:, idx1, idx2].reshape(Cin * K * K, Hout * Wout)
    # 矩阵相乘得到卷积后的结果
    res = (filter @ img).reshape(Cout, Hout, Wout)
    return res

if __name__ == '__main__':
    t.manual_seed(1000)

    # # 1.Tensor底层实现
    # # 改变形状,转置,切片  三个实例指向一个storeage
    # a = t.arange(6).view(2,3)
    # b = a.reshape(2,3)
    # c = a.transpose(1,0)
    # d = a[:2,1]
    # # print(id(a.storage()) == id(b.storage()), id(a.storage())== id(c.storage()), id(a.storage())== id(d.storage()))
    # print(id(a.untyped_storage()) == id(b.untyped_storage()), id(a.untyped_storage()) == id(c.untyped_storage()),id(a.untyped_storage()) == id(d.untyped_storage()))
    # """
    # False False False   a.storage()已经废弃
    # True True True      a.untyped_storage() √
    # """
    #
    # # 发生改变的是三个内部属性 size storage_offset stride
    # print(a.size(),a.storage_offset(),a.stride())
    # print(b.size(),b.storage_offset(),b.stride())
    # print(c.size(),c.storage_offset(),c.stride())
    # print(d.size(),d.storage_offset(),d.stride())
    # """
    # torch.Size([2, 3]) 0 (3, 1)
    # torch.Size([2, 3]) 0 (3, 1)
    # torch.Size([3, 2]) 0 (1, 3)
    # torch.Size([2]) 1 (3,)
    # """
    #
    # a = t.arange(12).view(3,4)
    # print(a,a.size(),a.storage_offset(),a.stride())
    # """
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])
    # torch.Size([3, 4])
    # 0
    # (4, 1)
    # """
    #
    # a = t.randn(2,3,4,5)
    # print(a.stride())
    # """
    # (60, 20, 5, 1)
    # """
    #
    # # 2.整数数组索引
    # a = t.arange(12).reshape(3, 4)
    # print(a)
    # print(a[t.tensor([1, 2]), t.tensor([0, 2])])    # 获取索引为[1,0] 与 [2,2]的元素
    # """
    # tensor([[ 0,  1,  2,  3],
    #         [ 4,  5,  6,  7],
    #         [ 8,  9, 10, 11]])
    # tensor([ 4, 10])
    # """
    #
    # print(a[t.tensor([1,2])[None,:], t.tensor([0, 2])[:,None]]) # 获取索引为[1,0],[2,0],[1,2],[2,2]的元素
    # """
    # tensor([[ 4,  8],
    #         [ 6, 10]])
    # """
    #
    # a = t.arange(24).view(2,3, 4)
    # index1 = t.tensor([1,0])
    # index2 = t.tensor([0,2])
    # value = a[:, index1, index2].shape
    # print(value)
    # print(a.shape[0], index1.shape)
    # """
    #
    # """


    # 3.布尔数组索引
    a = t.arange(12).view(3, 4)
    b = t.rand(3, 4)
    index_bool = b > 0.5     # 构建3*4的布尔型张量.
    print(a)
    print(b)
    print(index_bool)
    print(a[index_bool])
    """
    tensor([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]])
    tensor([[0.3189, 0.6136, 0.4418, 0.2580],
            [0.2724, 0.6261, 0.4410, 0.3653],
            [0.3535, 0.5971, 0.3572, 0.4807]])       
    tensor([[False,  True, False, False],
            [False,  True, False, False],
            [False,  True, False, False]]) 
    tensor([1, 5, 9])
    """

    # 布尔数组索引用于对特定条件下的数值进行修改
    # demo1:Tensor中>0的值扩大2倍
    a = t.tensor([[1, -3, 2], [2, 9, -1], [-8, 4, 1]])
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if a[i, j] > 0:
                a[i, j] *= 2
    print(a)

    a = t.tensor([[1, -3, 2], [2, 9, -1], [-8, 4, 1]])
    a[a > 0] *= 2
    print(a)
    """
    tensor([[ 2, -3,  4],
            [ 4, 18, -1],
            [-8,  8,  2]])
    tensor([[ 2, -3,  4],
            [ 4, 18, -1],
            [-8,  8,  2]])
    """

    # demo2:返回Tensor中所有行和小于3的行
    a = t.tensor([[1, -3, 2], [2, 9, -1], [-8, 4, 1]])
    row_sum = a.sum(-1)
    sum = a[row_sum < 3, :]
    print(row_sum)
    print(sum)
    """
    tensor([ 0, 10, -3])
    tensor([[ 1, -3,  2],
            [-8,  4,  1]])
    """

    # 4.用高级索引实现卷积
    img = t.arange(36).view(1, 6, 6)
    filters = t.ones(1, 1, 3, 3) / 9
    stride, padding = t.tensor(1.), t.tensor(0)
    output = Conv_base(img, filters, stride, padding)
    print("进行卷积操作的图像为:", img[0])
    print("卷积核为:", filters[0][0])
    print("卷积后的结果为:", output)
    """
    进行卷积操作的图像为: 
    tensor([[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
            [24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35]])
    卷积核为: 
    tensor([[0.1111, 0.1111, 0.1111],
            [0.1111, 0.1111, 0.1111],
            [0.1111, 0.1111, 0.1111]])
    卷积后的结果为: 
    tensor([[[ 7.0000,  8.0000,  9.0000, 10.0000],
             [13.0000, 14.0000, 15.0000, 16.0000],
             [19.0000, 20.0000, 21.0000, 22.0000],
             [25.0000, 26.0000, 27.0000, 28.0000]]])    
    """

    Cin, Hin, Win = 1, 6, 6
    img = t.arange(Cin*Hin*Win).view(Cin, Hin, Win).float()
    # 定义卷积核的大小和输出通道数
    K, Cout, stride = 3, 1, 1
    filter = t.ones((Cin, Cout, K, K)).float() / (Cin * K * K)
    conv = Conv(img, filter, stride)
    print(conv)
    """
    tensor([[[ 7.0000,  8.0000,  9.0000, 10.0000],
             [13.0000, 14.0000, 15.0000, 16.0000],
             [19.0000, 20.0000, 21.0000, 22.0000],
             [25.0000, 26.0000, 27.0000, 28.0000]]])
    """

    # 利用卷积操作对一张图像进行平均池化
    img = Image.open('./imgs/input.png')
    img.show()

    img = t.tensor(np.array(img)).float()
    img = img.transpose(0, 2).transpose(1, 2)     # 将img的形状从h×w×c转化为c×h×w

    Cin, Cout, K,  = 3, 3, 3                                             # 初始化卷积核 3*3*3*3
    filter_pool = t.zeros(Cin, Cout, K, K)
    filter_pool[t.arange(Cin), t.arange(Cin), :, :] = 1. / K / K         # 平均池化(对角线位置赋值)
    print(filter_pool, filter_pool.size())
    
    out = Conv(img, filter_pool, stride=K)              # 利用卷积去模拟池化,将步长设置为卷积核的大小即可
    out = out.transpose(1, 0).transpose(1, 2).long()    # 将输出结果转换为h×w×c的形状,用于显示
    Image.fromarray(np.array(out, dtype=np.uint8)).show()
    """
    filter_pool[0,0,:,:], filter_pool[1,1,:,:], filter_pool[2,2,:,:]...
    tensor([[[[0.1111, 0.1111, 0.1111],
              [0.1111, 0.1111, 0.1111],
              [0.1111, 0.1111, 0.1111]],
             [[0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]],
             [[0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]]],
            [[[0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]],
             [[0.1111, 0.1111, 0.1111],
              [0.1111, 0.1111, 0.1111],
              [0.1111, 0.1111, 0.1111]],
             [[0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]]],
            [[[0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]],
             [[0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000],
              [0.0000, 0.0000, 0.0000]],
             [[0.1111, 0.1111, 0.1111],
              [0.1111, 0.1111, 0.1111],
              [0.1111, 0.1111, 0.1111]]]]) torch.Size([3, 3, 3, 3])
    """

