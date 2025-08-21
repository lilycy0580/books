# coding: utf-8
import sys, os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)  # 所有照片是一维/一列Numpy数组形式保存
    img = x_train[0]
    label = t_train[0]
    img_reshape = img.reshape(28, 28)
    print(img.shape, label, img_reshape.shape)  # 5   (784,)    (28, 28)
    img_show(img_reshape)
