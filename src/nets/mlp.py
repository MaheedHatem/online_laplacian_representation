import torch
import torch.nn as nn
import numpy as np

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module
    
class MLP(nn.Module):
    def __init__(self, in_dim, sizes, activation=nn.ReLU(), output_activation=nn.Identity(), output_layer_init = np.sqrt(2)):
        super().__init__()
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2)
        )
        output_init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), output_layer_init
        )
        #layers = [nn.Flatten()]
        layers = []
        sizes = [np.product(in_dim)] + sizes
        for j in range(len(sizes)-1):
            if j < len(sizes) - 2:
                layers += [init_(nn.Linear(sizes[j], sizes[j+1])), activation]
            else:
                layers += [output_init_(nn.Linear(sizes[j], sizes[j+1])), output_activation]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)