from .encoder import Encoder
from typing import Tuple
import torch.nn as nn
import torch

class TabularRepresentation(Encoder):
    def __init__(self, dimension: int, obs_dimension, **kwargs):
        super().__init__(dimension, obs_dimension, **kwargs)
        assert len(obs_dimension) == 2
        self.W = nn.Parameter(torch.zeros(obs_dimension[0], obs_dimension[1], self.d, dtype=torch.float32, requires_grad=True))
        nn.init.uniform_(self.W, -2, 2)
    
    def forward(self, obs):
        assert len(obs.shape) == 2
        encoding = self.W[obs[:, 0], obs[:, 1]]
        assert encoding.shape[0] == obs.shape[0]
        assert encoding.shape[1] == self.d
        return encoding