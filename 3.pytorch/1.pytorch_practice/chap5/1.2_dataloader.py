

import os
import random
import torch as t
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import default_collate  # 导入默认的拼接方式
from torch.utils.data.sampler import  WeightedRandomSampler

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

# 自定义数据集,可处理异常样本数据
class NewDogCat(DogCat): 
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)   # 调用父类的获取函数 即 DogCat.__getitem__(self, index)
        except:
            return None, None

# 样本损坏或数据集加载异常,随机取一张图片代替出现异常的图片
class NewDogCat_(DogCat):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except:
            new_index = random.randint(0, len(self) - 1)
            return self[new_index]


# 过滤为图像为None的数据
def my_collate_fn(batch):                       # batch是一个list,每个元素是dataset的返回值,形如(data, label)
    batch = [_ for _ in batch if _[0] is not None]
    if len(batch) == 0:
        return t.Tensor(), t.Tensor()           # 返回空数据与空标签
    return default_collate(batch)               # 用默认方式拼接过滤后的batch数据

class BadDataset:
    def __init__(self):
        self.idxs = [] # 取数据的次数
    def __getitem__(self, index):
        self.idxs.append(index)
        return self.idxs
    def __len__(self):
        return 9



if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.数据集中无异常图片
    transform = transforms.Compose([                                    # 数据增强操作
             transforms.RandomResizedCrop(224),                         # 随机裁剪并调整大小
             transforms.RandomHorizontalFlip(p=1),                      # 水平翻转 p为概率
             transforms.ToTensor(),
             transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]), # 标准化
    ])
    dataset = ImageFolder('data/dogcat_2/', transform=transform)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, num_workers=0, drop_last=False)
    dataiter = iter(dataloader)
    imgs, labels = next(dataiter)
    print(imgs.size())     # batch_size, channel, height, width
    """
    torch.Size([3, 3, 224, 224])
    """

    for batch_datas, batch_labels in dataloader:
        # train()
        pass

    dataiter = iter(dataloader)                     # DataLoader是一个可迭代(iterable)对象,可以像使用迭代器一样使用它
    batch_datas, batch_labels = next(dataiter)
    print(batch_datas.size(),batch_labels.size(),batch_labels)
    """
    torch.Size([3, 3, 224, 224])            data
    torch.Size([3]) tensor([1, 1, 0])       label
    """

    # 2.样本损坏或数据集加载异常
    # 方式一:将异常图片设置为None并过滤掉
    dataset = NewDogCat('./data/dogcat_wrong/', transforms=transform)
    print(dataset[9])
    dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=0, shuffle=False)
    for batch_datas, batch_labels in dataloader:
        print(batch_datas.size(), batch_labels.size(), batch_labels)
    """
    (None, None)
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 1])
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 1])
    torch.Size([1, 3, 224, 224]) torch.Size([1]) tensor([0])        异常图片    
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([0, 0])
    torch.Size([1, 3, 224, 224]) torch.Size([1]) tensor([0])        异常图片
    """

    # 方式二:随机选择一张图片替换掉
    dataset = NewDogCat_('./data/dogcat_wrong/', transforms=transform)
    print(dataset[9][0].shape)
    dataloader = DataLoader(dataset, 2, collate_fn=my_collate_fn, num_workers=0, shuffle=False)
    for batch_datas, batch_labels in dataloader:
        print(batch_datas.size(), batch_labels.size(), batch_labels)
    """
    torch.Size([3, 224, 224])
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 1])
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 1])
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([1, 0])     异常图片    
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([0, 0])
    torch.Size([2, 3, 224, 224]) torch.Size([2]) tensor([0, 1])     异常图片  
    """

    # 3.Dataloader 多进程
    dataset = BadDataset()
    dataLoader = DataLoader(dataset, num_workers=4)
    print('start')
    for item in dataLoader:
        print(item)                      # 注意这里self.idxs的数值
    print('end')
    print('index of main', dataset.idxs)  # 注意这里的idxs和__getitem__返回的idxs的区别
    """
    start
    [tensor([0])]
    [tensor([1])]
    [tensor([2])]
    [tensor([3])]
    [tensor([0]), tensor([4])]
    [tensor([1]), tensor([5])]
    [tensor([2]), tensor([6])]
    [tensor([3]), tensor([7])]
    [tensor([0]), tensor([4]), tensor([8])]
    end
    index of main []
    """

    # 4.Sampler模块
    dataset = DogCat('data/dogcat/', transforms=transform)
    weights = [2 if label == 1 else 1 for data, label in dataset]   # p_dog:p_cat = 2:1
    print(weights)                                                  # 猫和狗图片被取出的概率与weights的绝对大小无关,只和比值有关
    """
    [2, 2, 2, 2, 1, 1, 1, 1]
    """

    sampler = WeightedRandomSampler(weights,num_samples=9,replacement=True)
    dataloader = DataLoader(dataset,batch_size=3,sampler=sampler)
    for datas, labels in dataloader:
        print(labels.tolist())
    """
    [0, 1, 0]
    [0, 1, 1]
    [1, 0, 0]    
    replacement=True:   
        猫狗样本比例约为1：2
        一共只有8个样本,但是却返回了9个,说明有样本被重复返回
    """

    sampler = WeightedRandomSampler(weights, 8, replacement=False)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    for datas, labels in dataloader:
        print(labels.tolist())
    """
    [0, 1, 1, 0]
    [0, 1, 1, 0]
    replacement=False:   
        num_samples等于dataset的样本总数. 
        为了不重复选取,Sampler会将每个样本都返回,weight参数不再生效        
    """
