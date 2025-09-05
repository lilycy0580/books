import torch as t
import torch.nn as nn
import numpy as np
import torchvision
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class ToyModel(nn.Module):
    def __init__(self, input_size=28, hidden_size=500, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

"""
安装TensorBoard
	conda activate learn_pytorch2 
	pip install tensorboard
	(learn_pytorch2) C:\Users\83584>cd D:\books\3.pytorch\1.pytorch_practice\chap5
	(learn_pytorch2) C:\Users\83584>d:
	(learn_pytorch2) D:\books\3.pytorch\1.pytorch_practice\chap5>tensorboard --logdir=./visual
	http://localhost:6006/
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    logger = SummaryWriter(log_dir='./visual')  # 构建logger对象,log_dir用来指定log文件的保存路径

    # 1.使用add_scalar记录标量
    for n_iter in range(100):
        logger.add_scalar('Loss/train', np.random.random(), n_iter)
        logger.add_scalar('Loss/test', np.random.random(), n_iter)
        logger.add_scalar('Acc/train', np.random.random(), n_iter)
        logger.add_scalar('Acc/test', np.random.random(), n_iter)

    # 2.使用add_image显示图像
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST('./data', download=True, train=False, transform=transform)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=16)
    images, labels = next(iter(dataloader))
    grid = torchvision.utils.make_grid(images)
    logger.add_image('images', grid, 0)

    # 3.使用add_graph可视化网络
    model = ToyModel()
    logger.add_graph(model, images)

    # 4.使用add_histogram显示直方图
    logger.add_histogram('normal', np.random.normal(0, 5, 1000), global_step=1)
    logger.add_histogram('normal', np.random.normal(1, 2, 1000), global_step=10)

    # 5.使用add_embedding可视化embedding
    dataset = datasets.MNIST('./data', download=True, train=False)
    images = dataset.data[:100].float()
    label = dataset.targets[:100]
    features = images.view(100, 784)
    logger.add_embedding(features, metadata=label, label_img=images.unsqueeze(1))

