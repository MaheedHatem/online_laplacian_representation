from abc import ABC, abstractmethod
import collections
import gymnasium as gym
from src.agents.agent import Agent
import numpy as np 
from src.nets import MLP
from copy import deepcopy
import torch
import torch.nn as nn

class DQN(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_network = MLP(self.obs_dim, self.hidden_dims+[self.num_actions]).to(self.device)
        self.target_network = deepcopy(self.q_network).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), self.lr)
        self.epsilon = self.init_epsilon
        self.random_steps = self.random_steps_fraction * self.total_train_steps
        self.update_target = False
    
    def act(self, state, env = None, deterministic = False):
        if self.rng.random() > self.epsilon or deterministic:
            with torch.no_grad():
                if isinstance(state, dict):
                    state = self.replay_buffer.get_state(state)
                obs = torch.as_tensor(state, dtype=torch.float32).to(self.device)
                q_val = self.q_network(obs)
                return torch.argmax(q_val, axis=-1).cpu().detach().numpy()
        return self.rng.integers(self.num_actions)

    def train(self):
        for _ in range(self.training_batches):
            batch = self.replay_buffer.get_batch(self.batch_size)
            batch = tuple(torch.as_tensor(data) for data in batch)
            obs, act, next_obs, rewards, done = batch
            batch_idx = torch.arange(len(act)).long()
            target_q = self.get_targets(next_obs, rewards, done)
            self.optimizer.zero_grad()
            loss = ((self.q_network(obs)[batch_idx, act]  - target_q)**2)
            loss = loss.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_network.parameters(), self.max_grad_norm)
            self.optimizer.step()
            if self.update_target:
                self.update_target = False
                with torch.no_grad():
                    for p, p_targ in zip(self.q_network.parameters(), self.target_network.parameters()):
                        p_targ.data.copy_((1-self.target_update)* p_targ.data + 
                            self.target_update * p.data)
                
    def get_targets(self, next_obs, rewards, done):
        with torch.no_grad():
            return rewards + (1 - done) * self.gamma * self.target_network(next_obs).max(dim=-1).values
        
    def update_hyperparameters(self, step):
        self.epsilon = max(self.final_epsilon, self.init_epsilon + step * (self.final_epsilon - self.init_epsilon)/self.random_steps)
        if step % self.target_update_every:
            self.update_target = True
        
    def get_policy(self, states):
        policy = np.ones((len(states), self.num_actions)) * self.epsilon / self.num_actions
        states_array = np.zeros((len(states),2))
        for (row, col), state in states.items():
            states_array[state] = np.array([row, col])
        acts = self.act(states_array, deterministic=True)
        for (row, col), state in states.items():
            policy[state, acts[state]] += (1- self.epsilon)
        return policy
    
    def save(self, save_dir, step):
        torch.save(self.q_network.state_dict(), f"{save_dir}/networks/q{step}.pth")

    def load(self, save_dir, step):
        self.q_network.load_state_dict(torch.load(f"{save_dir}/networks/q{step}.pth"))