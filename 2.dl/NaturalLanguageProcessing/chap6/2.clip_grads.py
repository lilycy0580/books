import numpy as np

if __name__ == '__main__':
    dW1 = np.random.rand(3, 3) * 10
    dW2 = np.random.rand(3, 3) * 10
    grads = [dW1, dW2]                          # g
    max_norm = 5.0                              # 阈值


    def clip_grads(grads, max_norm):
        total_norm = 0
        for grad in grads:
            total_norm += np.sum(grad ** 2)
        total_norm = np.sqrt(total_norm)        # L2范数 开根号

        rate = max_norm / (total_norm + 1e-6)   # 梯度裁剪
        if rate < 1:
            for grad in grads:
                grad *= rate

    print('before:', dW1.flatten())
    clip_grads(grads, max_norm)
    print('after:', dW1.flatten())
    # before:[0.25807993 6.32817793 1.75347082 2.09374069 7.1831121  2.96525077 0.58708432 6.68073053 6.97974796]
    # after: [0.0554999  1.36086991 0.37708258 0.45025736 1.54472286 0.63767495 0.12625204 1.43668608 1.50098955]



