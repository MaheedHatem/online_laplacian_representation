import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal
from .mlp import MLP

class GaussianModel(MLP):
    def __init__(self, in_dim, sizes, activation=nn.Tanh(), output_activation=nn.Identity(), output_layer_init = np.sqrt(2)):
        super().__init__(in_dim, sizes, activation, output_activation, output_layer_init)
        log_std = np.zeros( sizes[-1], dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        
    
    def get_distribution(self, x):
        mu = self.model(x)
        action_logstd = self.log_std.expand_as(mu)
        action_std = torch.exp(action_logstd)
        return Normal(mu, action_std)

    def get_act_prob(self, x, act):
        dist = self.get_distribution(x)
        return dist.log_prob(act).sum(axis=-1) , dist.entropy().sum(axis=-1).mean()