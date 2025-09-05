
import torch as t

if __name__ == '__main__':
    """
    Chap5.Pytorch中常用的工具
        1.数据处理
            Dataset
            DataLoader
        2.预训练模型
        3.可视化工具
            TensorBoard
            Visdom
        4.使用GPU加速:CUDA    
    """
    """
    1.数据处理
        深度学习需花费大量精力处理数据,包括图像,文本,语音或其他二进制数据等
        
        1.Dataset
            数据加载通过自定义数据集对象实现,数据集对象被抽象为Dataset类,自定义数据集需继承Dataset
                __getitem__() 返回一条数据或一个样本 obj[index]等价于obj.__getitem__(index)
                __len__()     返回样本的数量        len(obj)等价于obj.__len__()
            
            torchvision视觉工具包,提高视觉图像处理工具,transforms模块提供一系列数据增强的操作
                1.torchvision.transforms
                    仅支持PILImage对象的操作:
                        RandomChoice        在一系列transforms操作中随机执行一个操作
                        RandomOrder         以随意顺序执行一系列transforms操作
                    仅支持Tensor对象的操作:
                        Normalize           标准化，即减去均值，除以标准差
                        RandomErasing       随机擦除Tensor中一个矩形区域的像素
                        ConvertImageDtype   将Tensor转换为指定的类型，并进行相应的缩放
                    PILImage对象与Tensor对象相互转换:
                        ToTensor
                        ToPILImage
                    既支持PILImage对象又支持Tensor对象:
                        Resize               调整图片尺寸
                        CenterCrop,RandomCrop,RandomResizedCrop,FiveCrop        按照不同规则对图像进行裁剪
                        RandomAffine         随机进行仿射变换,保持图像中心不变
                        RandomGrayscale      随机将图像变为灰度图
                        RandomHorizontalFlip,RandomVerticalFlip,RandomRotation  随机水平翻转、垂直翻转、旋转图像
                    对图像进行多种操作
                        transforms.Compose
                    通过Lambda封装自定义的转换策略
                        trans = transforms.Lambda(lambda img:img.rotate(random()*360))
                    
                2.torchvision.transforms.functional 
                    adjust_brightness,adjust_contrast       调整图像的亮度,对比度
                    crop,center_crop,five_crop,ten_crop     对图像按不同规则进行裁剪 
                    normalize                               标准化,即减均值,除以标准差
                    to_tensor                               将PILImage对象转成Tensor
                
                3.torchvision.datasets
                    常用数据集 CIFAR-10 ImageNet COCO MNIST LSUN
                    ImageFolder(root,transform=None,target_transform=None,loader=default_loader,is_valid_file=None)
                        root                在root路径下寻找图像
                        transform           对PILImage进行相关数据增强,输入为使用loader读取图像的返回对象
                        target_transform    对label进行转换
                        loader              指定加载图像的函数,默认操作是读取PIL Image对象
                        is_valid_file       获取图像路径,检查文件的有效性
                  
        2.DataLoader 
             Dataset只负责数据的抽象,调用一次__getattr__(),返回一个样本
             训练神经时,一次处理的对象是一个batch的数据,需要对一批数据进行打乱顺序和并行加速等操作  DataLoader
                DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, 
                           batch_sampler=None, num_workers=0, collate_fn=None, 
                           pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None, 
                           multiprocessing_context=None, generator=None, *, prefetch_factor=2, 
                           persistent_workers=False)
                    dataset         加载的数据集,Dataset对象
                    batch_size
                    shuffle
                    sampler         样本抽样
                    batch_sampler   一次返回一个batch的索引
                    num_workers     使用多进程加载的进程数，0代表不使用多进程
                    collate_fn      如何将多个样本数据拼接成一个batch，一般使用默认的拼接方式即可
                    pin_memory      是否将数据保存在pin memory区，pin memory中的数据转移到GPU速度更快
                    drop_last       dataset中的数据个数可能不是batch_size的整数倍，若drop_last为True，则将多出来不足一个batch的数据丢弃
                    timeout         进程读取数据的最大时间，若超时则丢弃数据
                    worker_init_fn  每个worker的初始化函数
                    prefetch_factor 每个worker预先加载的样本数
    """

    """
    
    """
    """

    """
    """

    """
    """

    """
    """

    """
    """

    """
    """

    """
