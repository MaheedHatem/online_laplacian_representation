from .encoder import Encoder
from typing import Tuple
import torch.nn as nn
import torch
from src.nets import MLP
class MLPEncoder(Encoder, nn.Module):
    def __init__(self, dimension: int, obs_dimension, **kwargs):
        Encoder.__init__(self, dimension, obs_dimension, **kwargs)
        nn.Module.__init__(self)
        if isinstance(obs_dimension, tuple):
            self.encoder = MLP(len(obs_dimension), self.hidden_dims + [self.d])
        else:
            self.encoder = MLP(obs_dimension, self.hidden_dims + [self.d])
    
    def forward(self, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        encoding = self.encoder(obs)
        assert encoding.shape[0] == obs.shape[0]
        assert encoding.shape[1] == self.d
        return encoding