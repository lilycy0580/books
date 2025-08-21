import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

if __name__ == '__main__':
    x = np.arange(0,6,0.1)     # [0,6] step =0.1
    y = np.sin(x)
    plt.plot(x,y)
    plt.show()
    plt.savefig('test1.png')

    x = np.arange(0,6,0.1)     # [0,6] step =0.1
    y1 = np.sin(x)
    y2 = np.cos(x)
    plt.plot(x,y1,label='sin')
    plt.plot(x,y2,label='cos')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('sin & cos')
    plt.legend()
    plt.show()
    plt.savefig('test2.png')

    img = imread('lena.png')
    print(img.ndim,img.shape)
    if img.ndim == 3:
        gray_img = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        plt.imshow(gray_img,cmap='gray')
    else:
        plt.imshow(img)
    plt.savefig('test3.png')
    plt.show()


