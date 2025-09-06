
import torch as t
import torch.nn as nn
from PIL import Image
from torchvision.transforms import ToTensor,ToPILImage

if __name__ == '__main__':
    t.manual_seed(1000)

    # 2.1.1卷积层
    to_tensor = ToTensor()      # image-->Tensor
    to_pil = ToPILImage()
    lena = Image.open('./lena.png')
    lena.show()
    """
    显示lena的灰色图片
    """

    lena = to_tensor(lena).unsqueeze(0)
    print(f'Input lena shape: {lena.shape}')
    """
    Input lena shape: torch.Size([1, 1, 200, 200])
    """

    # 锐化卷积核
    kernel = t.ones(3,3)/(-9.)
    print(kernel)
    kernel[1][1] = 1
    print(kernel)
    """
    tensor([[-0.1111, -0.1111, -0.1111],
            [-0.1111, -0.1111, -0.1111],
            [-0.1111, -0.1111, -0.1111]])
    tensor([[-0.1111, -0.1111, -0.1111],
            [-0.1111,  1.0000, -0.1111],
            [-0.1111, -0.1111, -0.1111]])
    """

    # 构建一个二维卷积层并手动设置权重  w_size = (out_channels, in_channels, kernel_height, kernel_width)
    conv = nn.Conv2d(1,1,(3,3),1,bias=False)
    conv.weight.data = kernel.view(1,1,3,3)

    out = conv(lena)
    print(f'Output shape: {out.shape}')
    """
    Output shape: torch.Size([1, 1, 198, 198])
    """

    to_pil(out.data.squeeze()).show()
    """
    显示lena的黑色图片
    """

    # 2.1.2.池化层
    input = t.randint(10,(1,1,4,4))     # 均匀分布[0,9] size=(1,1,4,4)
    print(input)
    """
    tensor([[[[4, 5, 5, 4],
              [3, 4, 3, 0],
              [4, 7, 1, 2],
              [8, 5, 0, 2]]]])
    """

    pool = nn.AvgPool2d(2,2)
    pool(input)
    print(pool)
    """
    AvgPool2d(kernel_size=2, stride=2, padding=0)
    """

    params = list(pool.parameters())
    print(params)
    """
    []
    """

    out = pool(lena)
    to_pil(out.data.squeeze()).show()
    """
    显示灰色的lena图像
    """

    # 2.1.3.其他层
    # Linear()层
    # 注意以下例子都是对module的可学习参数进行直接操作,实际使用中,这些参数一般会随着模型的学习不断改进,除非需要特殊初始化,否则不要直接修改超参数
    input = t.randn(2,3)        # 输入数据,batch_size=2,维度为3
    linear = nn.Linear(3,4)
    h = linear(input)
    print(h)
    """
    tensor([[ 0.2612,  0.1758,  0.1205, -0.4857],
            [-0.9922,  0.4737,  0.6357,  0.5892]], grad_fn=<AddmmBackward0>)
    """

    # BatchNorm
    bn = nn.BatchNorm1d(4)          # 输入数据,(batch_size, 4) 4表示特征数量,分别对4个特征进行归一化处理
    bn.weight.data = t.ones(4)*4    # [4, 4, 4, 4]
    bn.bias.data = t.zeros(4)       # [0, 0, 0, 0]

    bn_out = bn(h)
    print(bn_out,bn_out.mean(0),bn_out.std(0,unbiased=False))
    """
    tensor([[ 3.9999, -3.9991, -3.9997, -3.9999],
            [-3.9999,  3.9991,  3.9997,  3.9999]],grad_fn=<NativeBatchNormBackward0>)
    tensor([-1.1921e-07, -2.3842e-07,  0.0000e+00,  0.0000e+00],grad_fn=<MeanBackward1>) 
    tensor([3.9999, 3.9991, 3.9997, 3.9999], grad_fn=<StdBackward0>)
    """

    # Dropout
    dropout = nn.Dropout(0.5)   # 每个元素以0.5的概率随机舍弃
    o = dropout(bn_out)
    print(o)
    """
    tensor([[ 7.9999, -0.0000, -0.0000, -7.9999],
            [-0.0000,  7.9982,  0.0000,  7.9999]], grad_fn=<MulBackward0>)
    """






