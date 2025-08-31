
import torch as t
import numpy as np
import warnings

from sympy.codegen import Print

"""
    2.命名张量
    3.Tensor与Numpy
    4.Tensor的基本结构
"""
if __name__ == '__main__':
    t.manual_seed(1000)
    warnings.filterwarnings('ignore')

    # # 2.命名张量
    # img = t.randn(1, 2, 3, 3, names=('N','C','H','W'))
    # print(img.names)
    # """
    # ('N', 'C', 'H', 'W')
    # """
    #
    # img_rotate = img.transpose(2, 3)    # transpose():交换第2和第3维度
    # print(img_rotate.names)
    # """
    # ('N', 'C', 'W', 'H')
    # """
    #
    # another_img = t.rand(1,3,2,2)
    # another_img = another_img.refine_names('N',None,'H','W')
    # print(another_img.names)
    # """
    # ('N', None, 'H', 'W')
    # """
    #
    # rename_img = img.rename(H='height',W='width')
    # print(rename_img.names)
    # """
    # ('N', 'C', 'height', 'width')
    # """
    #
    # convert_img = rename_img.align_to('N','height','width','C')
    # print(convert_img.names)
    # """
    # ('N', 'height', 'width', 'C')
    # """
    #
    # a = t.randn(1,2,2,3, names=('N','C','H','W'))
    # b = t.randn(1,2,2,2, names=('N','H','C','W'))
    # print(a+b)
    # """
    # Traceback (most recent call last):
    # File "D:\books\3.pytorch\1.pytorch_practice\chap3\1.Tensor2.py", line 51, in <module>
    #     print(a+b)
    # RuntimeError: Error when attempting to broadcast dims ['N', 'C', 'H', 'W'] and dims ['N', 'H', 'C', 'W']:
    #     dim 'H' and dim 'C' are at the same position from the right but do not match.
    # """

    # 3.Tensor与Numpy
    # a = np.ones([2,3],dtype=np.float32)
    # print(a,a.dtype)
    # b = t.from_numpy(a)
    # print(b,b.dtype)
    # a[0,1] = -1
    # print(b,b.dtype)
    # """
    # [[1. 1. 1.]
    #  [1. 1. 1.]] float32
    # tensor([[1., 1., 1.],
    #         [1., 1., 1.]]) torch.float32
    # tensor([[ 1., -1.,  1.],
    #         [ 1.,  1.,  1.]]) torch.float32
    # """
    #
    # a = np.ones([2,3])
    # print(a.dtype)
    #
    # b = t.Tensor(a)
    # print(b.dtype)
    #
    # c = t.from_numpy(a)
    # print(c)
    #
    # a[0,1] = -1
    # print(a,'\n',b,'\n',c)
    # """
    # float64
    # torch.float32
    # tensor([[1., 1., 1.],
    #         [1., 1., 1.]], dtype=torch.float64)
    #
    # [[ 1. -1.  1.]
    #  [ 1.  1.  1.]]
    #  tensor([[1., 1., 1.],
    #         [1., 1., 1.]])
    #  tensor([[ 1., -1.,  1.],
    #         [ 1.,  1.,  1.]], dtype=torch.float64)
    # """
    #
    # a_tensor = t.tensor(a)
    # a_tensor[0,1] = 1
    # print(a,a_tensor)
    # """
    # [[ 1. -1.  1.]
    #  [ 1.  1.  1.]]
    #  tensor([[1., 1., 1.],
    #         [1., 1., 1.]], dtype=torch.float64)
    # """

    # 4.Tensor的基本结构
    # a = t.arange(0,6).float()
    # b = a.view(2,3)
    # print(a.storage().data_ptr() == b.storage().data_ptr())     # 获取Tensor的数据存储块的首地址(指针)
    # """
    # True
    # """
    #
    # a[1] = 100
    # print(a,'\n',b)
    # """
    # tensor([  0., 100.,   2.,   3.,   4.,   5.])
    # tensor([[  0., 100.,   2.],
    #         [  3.,   4.,   5.]])
    # """
    #
    # c = a[2:]
    # print(a.storage().data_ptr(),c.storage().data_ptr(),        # Tensor存储区的内存地址
    #       a.data_ptr(),c.data_ptr(),                            # Tensor首元素的内存地址
    #       a.storage().data_ptr() == c.storage().data_ptr())
    # """
    # 6550164930560 6550164930560
    # 6550164930560 6550164930568
    # True
    # """
    #
    # c[0] = -100         # c[0]对应a[2]
    # print(a)
    # """
    # tensor([   0.,  100., -100.,    3.,    4.,    5.])
    # """
    #
    # d = t.Tensor(c.storage())
    # d[0] = 666
    # print(b)
    # """
    # tensor([[ 666.,  100., -100.],
    #         [   3.,    4.,    5.]])
    # """
    #
    # print(a.storage_offset() ,c.storage_offset(),d.storage_offset())
    # print(a.storage().data_ptr() == b.storage().data_ptr() == c.storage().data_ptr() == d.storage().data_ptr())
    # """
    # 0 2 0
    # True
    # """
    #
    # e = b[::2,::2]
    # print(a.storage().data_ptr()==e.storage().data_ptr())
    # print(e.is_contiguous())
    # """
    # True
    # False
    # """

    # # 5.N种改变Tensor形状的方法
    # tensor = t.arange(24).reshape(2,3,4)
    # print(f'tensor.shape: {tensor.shape},tensor.size(): {tensor.size()}')
    # print(f'Tensor的维度: {tensor.dim()},共有{tensor.numel()}个元素')
    # """
    # tensor.shape: torch.Size([2, 3, 4]),tensor.size(): torch.Size([2, 3, 4])
    # Tensor的维度: 3,共有24个元素
    # """
    #
    # a = t.arange(1,13)
    # b = a.view(2,6)
    # c = a.reshape(2,6)
    # print(id(a)==id(b)==id(c))
    # print(a.storage().data_ptr()==b.storage().data_ptr()==c.storage().data_ptr())
    # """
    # False   此处view()与reshape()等价,因为Tensor是连续的,但a,b,c三者内存地址不同
    # True    view与reshape存储在与原始对象不同的内存中,但共享存储区
    # """
    #
    # b = b.t()       # b不在连续,reshape()可变形,但view()报错
    # b.reshape(-1,4)
    # b.view(-1,4)
    # """
    # Traceback (most recent call last):
    # File "D:\books\3.pytorch\1.pytorch_practice\chap3\1.Tensor2.py", line 186, in <module>
    #     b.view(-1,4)
    # RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
    # """
    #
    # img_3xHxW = t.randn(3,128,256)
    # img_3xHW = img_3xHxW.view(3,-1)     # img_3xHxW.flatten(1,2)
    # mean = img_3xHW.mean(dim=1)
    # print(mean)
    # """
    # tensor([ 0.0039, -0.0020,  0.0008])
    # """
    #
    # alpha_HxW = t.rand(128,256)
    # alpha_1xHxW = alpha_HxW[None]      # alpha_HxW.unsqueeze(0)
    # rgba_img = t.cat([alpha_1xHxW,img_3xHxW],dim=0)
    # print(alpha_1xHxW.shape,rgba_img.shape)
    # """
    # torch.Size([1, 128, 256])
    # torch.Size([4, 128, 256])
    # """
    #
    # alpha_HxW = alpha_1xHxW.view(128,256)     # alpha_1xHxW.squeeze(0)  alpha_1xHxW.flattem(0,1)(0)  alpha_1xHxW[0]
    # print(alpha_HxW.shape)
    # """
    # torch.Size([128, 256])
    # """

    mask = t.arange(6).view(2, 3)
    print(mask)
    """
    tensor([[0, 1, 2],
            [3, 4, 5]])
    """

    # Tensor转置会导致内存不连续,但是view不会 图像旋转90度,即将第一维与第二维交换
    mask1 = mask.transpose(0,1)         # mask.T  mask.t()  mask.permute(1,0)
    print(mask1,mask1.is_contiguous())
    """
    tensor([[0, 3],
            [1, 4],
            [2, 5]])    False
    """

    mask2 = mask.view(3,2)
    print(mask2,mask2.is_contiguous())
    """
    tensor([[0, 1],
            [2, 3],
            [4, 5]])    True
    """

    # pytorch中,图像一般存储为C×H×W
    img_3xHxW = t.randn(3,128,256)

    # opencv/numpy/skimage中,图像一般存储为H×W×C
    img_HxWx3 = img_3xHxW.permute(1,2,0)

    contingus =  img_HxWx3.is_contiguous()
    print(contingus)
    """
    False
    """

    img = img_HxWx3.reshape(-1)
    # img = img_HxWx3.view(-1)      # 报错
    print(img)
    """
    reshape不会报错,view会报错
    tensor([-0.5306, -0.8402,  0.7344,  ...,  0.7608,  1.7374,  1.0216])
    """

    # reshape与transpose的使用
    H,W = 4,5
    img_3xHxW = t.randn(3,H,W)
    img_3xHW = img_3xHxW.reshape(3,-1)     # 目标数据排列和输入一致,直接使用reshape

    img_3xWH = img_3xHxW.transpose(1,2).reshape(3,-1)  # 目标数据排列和输入不一致,先通过transpose(3,W,H),再变为(3,WH)
    img_3xWxH = img_3xWH.reshape(3,W,H)
