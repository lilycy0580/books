
import torch as t

if __name__ == '__main__':
    t.manual_seed(1000)

    a = t.ones(1,3)
    b = a.expand(3,3)
    c = a.repeat(3,3)
    print(f'原始存储占用:'+str(a.storage().size()))
    print(f'expand存储占用:'+str(b.storage().size()))
    print(f'repeat存储占用:'+str(c.storage().size()))
    """
    原始存储占用:3
    expand存储占用:3
    repeat存储占用:27
    """

    """
    自动广播法则:
        step1:
            a是2维的,b是3维,需a前面补1个维度
            a.unsqueeze(0) a.shape=(1,3,2) 
                           b.shape=(2,3,1) 
        step2:
            a和b在第一和第三个维度的形状不一样,利用广播法则扩展成(2,3,2)
            
    """
    a = t.ones(3,2)
    b = t.zeros(2,3,1)
    print((a+b).shape)

    """
    手动广播法则:
        view 
        expand
    """
    sum = a.view(1, 3, 2).expand(2, 3, 2) + b.expand(2, 3, 2)
    sum = a[None, :, :].expand(2, 3, 2) + b.expand(2, 3, 2)       # 两种方式等价
    print(sum.shape)
    """
    torch.Size([2, 3, 2])
    torch.Size([2, 3, 2])
    """





