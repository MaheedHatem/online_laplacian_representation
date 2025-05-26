import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from .mlp import MLP

class CategoricalModel(MLP):
    def __init__(self, in_dim, sizes, activation=nn.ReLU(), output_activation=nn.Identity(), output_layer_init = np.sqrt(2)):
        super().__init__(in_dim, sizes, activation, output_activation, output_layer_init)
        
    
    def get_distribution(self, x):
        logits = self.model(x)
        return Categorical(logits=logits)

    def get_act_prob(self, x, act):
        dist = self.get_distribution(x)
        return dist.log_prob(act), dist.entropy().mean()