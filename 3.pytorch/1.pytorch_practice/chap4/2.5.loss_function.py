
import torch as t
import torch.nn as nn

if __name__ == '__main__':
    t.manual_seed(1000)

    score = t.randn(3,2)
    label = t.Tensor([1,0,1]).long()

    criterion = nn.CrossEntropyLoss()

    loss = criterion(score, label)
    print(loss)
    """
    tensor(1.3795)
    """