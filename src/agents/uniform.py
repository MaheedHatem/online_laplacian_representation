from abc import ABC, abstractmethod
import collections
import gymnasium as gym
from src.agents.agent import Agent
import numpy as np 

class UniformRandomPolicy(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    
    def act(self, state, env, deterministic=False):
        return env.action_space.sample()

    def train(self):
        pass
    
    def get_policy(self, states):
        return np.ones((len(states), self.num_actions)) / self.num_actions
    
    def update_hyperparameters(self, step):
        pass
    
    def save(self, save_dir, step):
        pass

    def load(self, save_dir, step):
        pass