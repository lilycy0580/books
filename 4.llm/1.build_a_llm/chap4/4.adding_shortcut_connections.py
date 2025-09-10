
import torch as t
import torch.nn as nn

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + t.tanh(t.sqrt(t.tensor(2.0 / t.pi)) *  (x + 0.044715 * t.pow(x, 3))))

def print_gradients(model, x):
    output = model(x)
    target = t.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)
    loss.backward()             # 仅计算梯度      optimizer.step()更新参数

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")  # 查看每层的梯度值

# 5层的神经网络,每层由一个线性层和一个GELU激活函数构成,当self.use_shortcut=True,添加快捷连接
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:     # self.use_shortcut=True,添加快捷连接
                x = x + layer_output
            else:
                x = layer_output
        return x


"""
    快捷连接
"""
if __name__ == '__main__':
    t.manual_seed(123)

    # a:没有快捷连接的神经网络
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = t.tensor([[1., 0., -1.]])
    model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=False)
    print_gradients(model_without_shortcut, sample_input)
    """
    layers.0.0.weight has gradient mean of 0.00020173587836325169
    layers.1.0.weight has gradient mean of 0.0001201116101583466
    layers.2.0.weight has gradient mean of 0.0007152041653171182
    layers.3.0.weight has gradient mean of 0.001398873864673078
    layers.4.0.weight has gradient mean of 0.005049646366387606     最后一层到第一层的过程逐渐变小,称为梯度消失
    """

    # b:有快捷连接的神经网络(包含跳跃连接的模型)
    model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut=True)
    print_gradients(model_with_shortcut, sample_input)
    """
    layers.0.0.weight has gradient mean of 0.0014432319439947605
    layers.1.0.weight has gradient mean of 0.004846962168812752
    layers.2.0.weight has gradient mean of 0.0041389018297195435
    layers.3.0.weight has gradient mean of 0.00591512955725193
    layers.4.0.weight has gradient mean of 0.03265950828790665      最后一层的梯度仍然大于其他层,
                                                                    但是梯度值在逐渐接近第一层时趋于稳定,没有缩小到几乎消失的程度
    """