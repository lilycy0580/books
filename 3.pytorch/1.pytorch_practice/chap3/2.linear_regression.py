import torch as t
from IPython import display
from matplotlib import pyplot as plt


# 生成随机数据 y=2x+3+噪声
def get_fake_data(device, batch_size=8):
    x = t.randn(batch_size, 1, device=device) * 5
    y = x * 2 + 3 + t.randn(batch_size, 1, device=device)
    return x, y


if __name__ == '__main__':
    t.manual_seed(1000)

    device = t.device('cuda:0' if t.cuda.is_available() else 'cpu')
    x, y = get_fake_data(device, batch_size=16)
    plt.scatter(x.squeeze().cpu().numpy(), y.squeeze().cpu().numpy())
    plt.show()
    plt.savefig('./fake_data.png')

    # 随机初始化参数
    w = t.rand(1, 1, device=device)
    b = t.zeros(1, 1, device=device)
    lr = 0.02
    for i in range(500):
        x, y = get_fake_data(device, batch_size=4)  # x.shape, y.shape : torch.Size([4, 1]) torch.Size([4, 1])

        # 前向传播
        y_pred = x.mm(w) + b.expand_as(y)  # y_pred.shape: torch.Size([4, 1])
        loss = 0.5 * (y_pred - y) ** 2
        loss = loss.mean()

        # 反向传播
        dloss = 1
        dy_pred = dloss * (y_pred - y)
        dw = x.t().mm(dy_pred)
        db = dy_pred.sum()

        # 更新参数
        w.sub_(lr * dw)
        b.sub_(lr * db)

        if i % 50 == 0:
            # 线性回归的结果
            x = t.arange(0, 6).float().view(-1, 1).to(device)
            y = x.mm(w) + b.expand_as(x)
            plt.plot(x.cpu().numpy(), y.cpu().numpy())

            # 真实数据
            x2, y2 = get_fake_data(device, batch_size=32)
            plt.scatter(x2.cpu().numpy(), y2.cpu().numpy())

            plt.xlim(0, 5)
            plt.ylim(0, 13)
            plt.show()
            plt.pause(0.5)
            plt.savefig('./linear_regression.png'.format(i))

    print(f'w:{w.item():.3f}, b:{b.item():.3f}')
    """
    w:1.981, b:2.955
    """


