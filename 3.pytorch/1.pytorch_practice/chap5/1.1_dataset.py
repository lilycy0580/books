
import torch as t
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision import transforms as transforms
import torchvision.transforms.functional as tf
import random
from torchvision.datasets import ImageFolder

# 自定义数据集 继承Dataset,实现两个方法
class DogCat_Old(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)                                 # 所有图片的绝对路径
        self.imgs = [os.path.join(root, img) for img in imgs]   # 此处不实际加载图片,仅指定路径,当调用__getitem__时才会读取img

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 1 if 'dog' in img_path.split('/')[-1] else 0    # dog->1, cat->0
        pil_img = Image.open(img_path)
        array = np.asarray(pil_img)
        data = t.tensor(array)
        return data, label

    def __len__(self):
        return len(self.imgs)

# 自定义数据集 torchvision.transforms
class DogCat(Dataset):
    def __init__(self, root, transforms=None):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, img) for img in imgs]
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = 0 if 'dog' in img_path.split('/')[-1] else 1
        data = Image.open(img_path)
        if self.transforms:
            data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.imgs)


# torchvision.transforms.functional 对多个对象以相同参数进行操作
# demo:随机旋转一对图像(将两种图像同时,同角度随机旋转)
def transforms_rotate(image1, image2):
    angle = random.randint(0, 360)
    image1 = tf.rotate(image1, angle)
    image2 = tf.rotate(image2, angle)
    return image1, image2

"""
Dataset
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.1.加载自定义数据集并查看对应的图片 DogCat
    dataset = DogCat_Old('./data/dogcat/')
    img, label = dataset[0]     # 相当于调用dataset.__getitem__(0)
    for img, label in dataset:
        print(img.size(), img.float().mean(), label)
    """
    torch.Size([500, 497, 3]) tensor(106.4917) 0
    torch.Size([499, 379, 3]) tensor(171.8088) 0
    torch.Size([236, 289, 3]) tensor(130.3022) 0
    torch.Size([374, 499, 3]) tensor(115.5157) 0
    torch.Size([375, 499, 3]) tensor(116.8187) 1
    torch.Size([375, 499, 3]) tensor(150.5085) 1
    torch.Size([377, 499, 3]) tensor(151.7140) 1
    torch.Size([400, 300, 3]) tensor(128.1548) 1
    """

    # 1.2.使用torchvision工具包对图片进行预处理
    transform = transforms.Compose([
        transforms.Resize(224), 		# 缩放图片(Image),保持长宽比不变,最短边为224像素
        transforms.CenterCrop(224), 	# 从图片中间切出224×224的图片
        transforms.ToTensor(),  		# 将图片(Image)转成Tensor,归一化至[0, 1]
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1],规定均值和标准差
    ])
    dataset = DogCat('./data/dogcat/', transforms=transform)
    img, label = dataset[0]
    for img, label in dataset:
        print(img.size(), label)
    """
    torch.Size([3, 224, 224]) 1
    torch.Size([3, 224, 224]) 1
    torch.Size([3, 224, 224]) 1
    torch.Size([3, 224, 224]) 1
    torch.Size([3, 224, 224]) 0
    torch.Size([3, 224, 224]) 0
    torch.Size([3, 224, 224]) 0
    torch.Size([3, 224, 224]) 0   
    """

    # 2.torchvision.transforms.functional
    pass

    # 3.torchvision.datasets
    dataset = ImageFolder('./data/dogcat_2/')
    print(dataset.class_to_idx)                 # cat label=0 dog label=1
    print(dataset.imgs)                         # 所有图片的路径和对应的label
    print(dataset[0][0],dataset[0][1])          # 返回PIL Image对象及其label
    dataset[0][0].show()
    """
    {'cat': 0, 'dog': 1}
    [('./data/dogcat_2/cat\\cat.12484.jpg', 0), 
     ('./data/dogcat_2/cat\\cat.12485.jpg', 0), 
     ('./data/dogcat_2/cat\\cat.12486.jpg', 0), 
     ('./data/dogcat_2/cat\\cat.12487.jpg', 0), 
     ('./data/dogcat_2/dog\\dog.12496.jpg', 1), 
     ('./data/dogcat_2/dog\\dog.12497.jpg', 1), 
     ('./data/dogcat_2/dog\\dog.12498.jpg', 1), 
     ('./data/dogcat_2/dog\\dog.12499.jpg', 1)]
    <PIL.Image.Image image mode=RGB size=497x500 at 0x1E30C853C10>  dataset[0][0] PIL Image对象
    0                                                               dataset[0][1] label
    """

    transform = transforms.Compose([                                    # 数据增强操作
             transforms.RandomResizedCrop(224),                         # 随机裁剪并调整大小
             transforms.RandomHorizontalFlip(p=1),                      # 水平翻转 p为概率
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]), # 标准化
    ])
    dataset = ImageFolder('data/dogcat_2/', transform=transform)
    size = dataset[0][0].size()                                         # 深度学习中,图像数据为CxHxW
    to_img = transforms.ToPILImage()
    img = dataset[0][0] * 0.2 + 0.4                                     # 0.2和0.4是标准差与均值的近似值
    to_img(img).show()