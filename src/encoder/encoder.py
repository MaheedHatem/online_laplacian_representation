from abc import ABC, abstractmethod
import torch.nn as nn
import torch
from typing import Tuple

class Encoder(ABC, nn.Module):
    def __init__(self, dimension: int, obs_dimension, **kwargs):
        super().__init__()
        self.d = dimension
        self.obs_dimension = obs_dimension
        self.__dict__.update((k, v) for k, v in kwargs.items())

    def save(self, save_dir, step):
        torch.save(self.state_dict(), f"{save_dir}/networks/encoder{step}.pth")

    def load(self, save_dir, step):
        self.load_state_dict(torch.load(f"{save_dir}/networks/encoder{step}.pth"))