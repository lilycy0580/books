
import torch as t
from torchvision import models
from torchvision import transforms as transforms
from PIL import Image
import numpy as np
import random
import cv2
from matplotlib import pyplot as plt

# 检测图像中的物体,然后在原图上可视化检测结果 (对mask与bounding box进行可视化)
def result(img_path, threshold=0.9, rect_th=1, text_size=1, text_th=2):
    """
    :param img_path: 输入图像路径
    :param threshold: 置信度阈值（只显示置信度 > 90% 的检测结果）
    :param rect_th: 边界框线条粗细
    :param text_size:文本大小
    :param text_th: 文本粗细
    :return:
    """
    masks, boxes = predict(img_path, threshold)     # 获取预测结果
    img = cv2.imread(img_path)                      # 使用opencv读取图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):                     # 遍历所有检测到的物体
        color_mask = color(masks[i])                # 生成彩色掩码
        img = cv2.addWeighted(img, 1, color_mask, 0.5, 0)                              # 叠加掩码到原图
        x1, y1 = boxes[i][0]  # 左上角
        x2, y2 = boxes[i][1]  # 右下角
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness=rect_th)    # 绘制边界框
    return img


# 图像处理并进行目标检测和实例分割的推理
def predict(img_path, threshold):
    """
    :param img_path:
    :param threshold: 置信度阈值（只保留置信度高于此值的检测结果）
    :return:
    """
    img = Image.open(img_path)                                      # 图像预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ])
    img = transform(img)

    pred = detection([img])                                         # 对图像进行预测(将图像输入模型中,获取输出结果)
    print(pred[0]['scores'].size())
    print(pred[0]['masks'].size())
    print(pred[0]['boxes'].size())
    print(pred[0]['labels'].size())
    """
    torch.Size([44])
    torch.Size([44, 1, 427, 640])
    torch.Size([44, 4])
    torch.Size([44])
    """
    score = list(pred[0]['scores'].detach().numpy())                # 提取置信度分数
    t = [score.index(x) for x in score if x > threshold][-1]
    print(score)
    print(t)
    """
    [np.float32(0.99630535), np.float32(0.98937654), np.float32(0.98879844), np.float32(0.9727304), np.float32(0.9724274), np.float32(0.93701833), np.float32(0.9180822), np.float32(0.90024877), np.float32(0.8640004), np.float32(0.8451458), np.float32(0.7945014), np.float32(0.73472923), np.float32(0.66413987), np.float32(0.6503798), np.float32(0.61393774), np.float32(0.5753154), np.float32(0.53003424), np.float32(0.47148648), np.float32(0.44014987), np.float32(0.3321948), np.float32(0.3039458), np.float32(0.284019), np.float32(0.28305617), np.float32(0.2630984), np.float32(0.20008895), np.float32(0.18686587), np.float32(0.15357672), np.float32(0.15031822), np.float32(0.11931365), np.float32(0.117710605), np.float32(0.11558466), np.float32(0.10852805), np.float32(0.107952364), np.float32(0.098421544), np.float32(0.09526591), np.float32(0.091925964), np.float32(0.08719366), np.float32(0.07722857), np.float32(0.074994735), np.float32(0.07333494), np.float32(0.06837963), np.float32(0.06735888), np.float32(0.058836095), np.float32(0.052495986)]
    7
    """

    mask = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()                                # 处理分割掩码
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]    # 边界框坐标
    print(mask.shape)
    print(pred_boxes)
    """
    (44, 427, 640)
    [[(np.float32(219.62843), np.float32(184.3034)), (np.float32(455.9712), np.float32(365.94092))], [(np.float32(306.97952), np.float32(116.63845)), (np.float32(373.97995), np.float32(150.28937))], [(np.float32(168.2403), np.float32(109.02191)), (np.float32(214.24432), np.float32(140.92697))], [(np.float32(190.8507), np.float32(107.604485)), (np.float32(290.2949), np.float32(156.05605))], [(np.float32(609.11945), np.float32(112.249146)), (np.float32(635.1915), np.float32(138.59045))], [(np.float32(293.82028), np.float32(116.16294)), (np.float32(336.3972), np.float32(149.99649))], [(np.float32(268.8827), np.float32(106.17355)), (np.float32(301.72223), np.float32(128.47522))], [(np.float32(480.51587), np.float32(109.269005)), (np.float32(524.43164), np.float32(131.12419))], [(np.float32(590.513), np.float32(107.70929)), (np.float32(618.1119), np.float32(127.383705))], [(np.float32(451.20718), np.float32(104.142395)), (np.float32(485.29385), np.float32(131.14343))], [(np.float32(80.41247), np.float32(87.859276)), (np.float32(119.85658), np.float32(108.160065))], [(np.float32(75.610306), np.float32(83.51096)), (np.float32(121.00774), np.float32(120.94296))], [(np.float32(287.2249), np.float32(105.886505)), (np.float32(327.74802), np.float32(127.29386))], [(np.float32(571.6003), np.float32(83.64131)), (np.float32(614.382), np.float32(124.85843))], [(np.float32(185.62776), np.float32(110.07753)), (np.float32(226.1484), np.float32(147.3414))], [(np.float32(99.50028), np.float32(89.05499)), (np.float32(118.64092), np.float32(101.36462))], [(np.float32(360.37375), np.float32(125.74905)), (np.float32(379.08087), np.float32(147.90791))], [(np.float32(135.95912), np.float32(91.52731)), (np.float32(185.95126), np.float32(126.084694))], [(np.float32(62.357067), np.float32(87.9033)), (np.float32(85.07043), np.float32(107.28992))], [(np.float32(569.46326), np.float32(107.24182)), (np.float32(592.52246), np.float32(127.88457))], [(np.float32(238.16043), np.float32(246.51271)), (np.float32(439.521), np.float32(351.9078))], [(np.float32(385.02963), np.float32(131.21455)), (np.float32(435.6206), np.float32(185.71768))], [(np.float32(254.19704), np.float32(179.85027)), (np.float32(460.8683), np.float32(291.39365))], [(np.float32(229.96703), np.float32(111.03033)), (np.float32(300.72308), np.float32(153.08833))], [(np.float32(268.85413), np.float32(103.869064)), (np.float32(328.41122), np.float32(147.58904))], [(np.float32(362.33926), np.float32(113.49285)), (np.float32(380.23825), np.float32(128.6433))], [(np.float32(53.31768), np.float32(108.33822)), (np.float32(73.075645), np.float32(129.2705))], [(np.float32(569.3608), np.float32(93.20682)), (np.float32(598.1783), np.float32(126.86351))], [(np.float32(262.90564), np.float32(117.47892)), (np.float32(301.38904), np.float32(148.73068))], [(np.float32(396.43024), np.float32(107.457344)), (np.float32(434.87753), np.float32(134.79669))], [(np.float32(134.62622), np.float32(89.28276)), (np.float32(156.2532), np.float32(105.673454))], [(np.float32(84.524475), np.float32(89.2585)), (np.float32(107.16528), np.float32(99.8437))], [(np.float32(178.93983), np.float32(103.85208)), (np.float32(301.1652), np.float32(155.48593))], [(np.float32(602.5333), np.float32(109.8371)), (np.float32(618.992), np.float32(126.983505))], [(np.float32(362.80215), np.float32(125.6187)), (np.float32(378.21414), np.float32(136.00731))], [(np.float32(267.42804), np.float32(186.2302)), (np.float32(447.7202), np.float32(286.87888))], [(np.float32(44.152084), np.float32(110.67631)), (np.float32(63.89571), np.float32(144.7164))], [(np.float32(391.50397), np.float32(133.63882)), (np.float32(432.84937), np.float32(179.94418))], [(np.float32(107.729866), np.float32(92.577675)), (np.float32(118.8876), np.float32(101.45376))], [(np.float32(134.65898), np.float32(89.48561)), (np.float32(145.67761), np.float32(104.3448))], [(np.float32(388.40237), np.float32(130.69214)), (np.float32(433.24704), np.float32(189.60672))], [(np.float32(389.45093), np.float32(122.76737)), (np.float32(430.90155), np.float32(187.68729))], [(np.float32(98.51531), np.float32(89.94828)), (np.float32(109.843765), np.float32(102.124886))], [(np.float32(482.5248), np.float32(109.9521)), (np.float32(516.4429), np.float32(123.57022))]]
    """
    # 返回过滤后的分割掩码和边界框
    pred_masks = mask[:t + 1]                                                                       # 过滤低置信度结果
    boxes = pred_boxes[:t + 1]
    return pred_masks, boxes

# 随机颜色,以便可视化
def color(image):
    colours = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]
    R = np.zeros_like(image).astype(np.uint8)
    G = np.zeros_like(image).astype(np.uint8)
    B = np.zeros_like(image).astype(np.uint8)
    R[image == 1], G[image == 1], B[image == 1] = colours[random.randrange(0, 3)]
    color_mask = np.stack([R, G, B], axis=2)
    return color_mask

"""
预训练模型
"""
if __name__ == '__main__':
    t.manual_seed(1000)

    # 1.使用torchvision.models加载预训练模型
    # 创建MobileNetV2模型(图像分类), 仅使用网络结构,参数权重随机初始化
    mobilenet_v2 = models.mobilenet_v2()
    print(mobilenet_v2)
    """
    MobileNetV2(
      (features): Sequential(
        (0): Conv2dNormActivation(
          (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
        (1): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
              (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
              (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (3): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (4): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
              (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (5): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (6): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (7): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
              (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (8): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (9): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (10): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (11): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
              (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (12): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (13): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (14): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
              (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (15): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (16): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (17): InvertedResidual(
          (conv): Sequential(
            (0): Conv2dNormActivation(
              (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (1): Conv2dNormActivation(
              (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
              (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU6(inplace=True)
            )
            (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (18): Conv2dNormActivation(
          (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU6(inplace=True)
        )
      )
      (classifier): Sequential(
        (0): Dropout(p=0.2, inplace=False)
        (1): Linear(in_features=1280, out_features=1000, bias=True)
      )
    )    
    """

    # 加载预训练的DeepLabV3模型(语义分割任务-->为图像中的每个像素分配类别标签),使用ResNet-50作为主干网
    deeplab = models.segmentation.deeplabv3_resnet50(pretrained=True)   # 加载预训练权重(COCO数据集上预训练的权重)
    print(deeplab)
    """
    DeepLabV3(
      (backbone): IntermediateLayerGetter(
        (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (relu): ReLU(inplace=True)
        (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        (layer1): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer2): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(256, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
              (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): Bottleneck(
            (conv1): Conv2d(512, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(128, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer3): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(512, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(512, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (3): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (4): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (5): Bottleneck(
            (conv1): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
            (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(256, 1024, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
        (layer4): Sequential(
          (0): Bottleneck(
            (conv1): Conv2d(1024, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2), dilation=(2, 2), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
            (downsample): Sequential(
              (0): Conv2d(1024, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            )
          )
          (1): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
          (2): Bottleneck(
            (conv1): Conv2d(2048, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(4, 4), dilation=(4, 4), bias=False)
            (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (conv3): Conv2d(512, 2048, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (bn3): BatchNorm2d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (relu): ReLU(inplace=True)
          )
        )
      )
      (classifier): DeepLabHead(
        (0): ASPP(
          (convs): ModuleList(
            (0): Sequential(
              (0): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (1): ASPPConv(
              (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(12, 12), dilation=(12, 12), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (2): ASPPConv(
              (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(24, 24), dilation=(24, 24), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (3): ASPPConv(
              (0): Conv2d(2048, 256, kernel_size=(3, 3), stride=(1, 1), padding=(36, 36), dilation=(36, 36), bias=False)
              (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (2): ReLU()
            )
            (4): ASPPPooling(
              (0): AdaptiveAvgPool2d(output_size=1)
              (1): Conv2d(2048, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
              (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
              (3): ReLU()
            )
          )
          (project): Sequential(
            (0): Conv2d(1280, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
            (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
            (3): Dropout(p=0.5, inplace=False)
          )
        )
        (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (3): ReLU()
        (4): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
      )
      (aux_classifier): FCNHead(
        (0): Conv2d(1024, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
        (3): Dropout(p=0.1, inplace=False)
        (4): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))
      )
    )
    """


    # demo:使用预训练好的实例分割模型Mask RCNN进行一次简单的实例分割
    # 1.加载预训练好的模型,不存在的话会自动下载  预训练好的模型保存在 ~/.torch/models/下面
    detection = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    detection.eval()        # 模型设置为评估模式:
                            #   (禁用dropout和batch normalization的训练特定行为 )
                            #   使用固定的统计量进行推理(而不是从当前批次计算)
    # 2.使用预训练的Mask R-CNN模型对输入图像进行目标检测和实例分割,然后将检测结果(边界框和分割掩码)可视化在图像上
    img = result('./data/demo.jpg')
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(img)
    plt.show()