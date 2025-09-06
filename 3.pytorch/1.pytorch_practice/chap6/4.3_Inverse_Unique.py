
import torch as t

if __name__ == '__main__':
    t.manual_seed(1000)

    a = t.randint(1,10,(10,15,10,5))                            # 4维的随机整数张量,每个元素范围[1,10) 10*15*10*5
    output, inverse_indices = t.unique(a, return_inverse=True)  # 获取输出的唯一列表和索引
    a_gen = output[inverse_indices]                             # 还原Tensor
    print(output)
    print(inverse_indices)
    print(a_gen.equal(a))
    """
    output:tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    inverse_indices:略
    True
    """


