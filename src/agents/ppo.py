from abc import ABC, abstractmethod
import collections
import gymnasium as gym
from src.agents.agent import Agent
import numpy as np 
from src.nets import MLP, CategoricalModel, GaussianModel
from copy import deepcopy
import torch
import torch.nn as nn
from itertools import chain

class PPO(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.critic = MLP(self.obs_dim, self.hidden_dims+[1]).to(self.device)
        if self.continuous_env:
            self.actor = GaussianModel(self.obs_dim, self.hidden_dims + [self.num_actions], output_layer_init=0.01).to(self.device)
        else:
            self.actor = CategoricalModel(self.obs_dim, self.hidden_dims + [self.num_actions], output_layer_init=0.01).to(self.device)
        self.optimizer = torch.optim.Adam(chain(self.critic.parameters(), self.actor.parameters()), self.lr, eps = self.adam_eps)
        self.replay_buffer.get_value = self.get_value
        self.replay_buffer._gamma = self.gamma
        self.clip_ratio = self.init_clip_ratio
        self.clipping_steps = 0.5 * self.total_train_steps
    
    def act(self, state, env=None, deterministic=False):
        with torch.no_grad():
            if isinstance(state, dict):
                state = self.replay_buffer.get_state(state)
            obs = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            dist = self.actor.get_distribution(obs)
            return dist.sample().detach().cpu().numpy()
        
    def get_value(self, state):
        with torch.no_grad():
            if isinstance(state, dict):
                state = self.replay_buffer.get_state(state)
            obs = torch.as_tensor(state, dtype=torch.float32).to(self.device)
            return self.critic(obs).detach().cpu().numpy()

    def normalize(self, adv):
        std = torch.std(adv) + 1e-8
        return (adv - torch.mean(adv))/std

    def critic_loss(self, obs, ret):
        adv = ret - self.critic(obs).squeeze()
        loss = 0.5 * ((adv)**2).mean()
        return loss
    

    def actor_loss(self, obs, act, old_prob, advantages):
        act_prob, entropy = self.actor.get_act_prob(obs, act)
        ratio = torch.exp(act_prob - old_prob)
        if (self.clip):
            clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
            loss = -(torch.min(ratio * advantages, clip_adv)).mean()
        else:
            loss = -(ratio * advantages).mean()
        return loss, entropy

    def train(self):
        self.compute_additional_data()
        data = self.replay_buffer.get_data(self.use_gae, self.gae_lambda)
        data = tuple(torch.as_tensor(d).to(self.device) for d in data)
        obs, act, _, ret, adv = data
        with torch.no_grad():
            old_prob, _ = self.actor.get_act_prob(obs, act)
        n_batches = len(obs) // self.batch_size
        for _ in range(self.training_batches):
            b_inds = torch.randperm(len(obs))
            for i in range(n_batches):
                mb_inds = b_inds[i * self.batch_size: (i+1)*self.batch_size]
                self.optimizer.zero_grad()
                critic_loss = self.critic_loss(obs[mb_inds], ret[mb_inds])
                mb_adv = self.normalize(adv[mb_inds])
                actor_loss, entropy = self.actor_loss(obs[mb_inds], act[mb_inds], old_prob[mb_inds], mb_adv)
                loss = self.val_coef * critic_loss + actor_loss - self.entropy_coef * entropy
                loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def compute_additional_data(self):
        pass
        
    def update_hyperparameters(self, step):
        if step >= self.clipping_steps:
            self.clip_ratio = max(self.final_clip_ratio, self.init_clip_ratio + (step-self.clipping_steps) * (self.final_clip_ratio - self.init_clip_ratio)/self.clipping_steps)
        if self.lr_annealing:
            frac = 1.0 - (step) / self.total_train_steps
            self.optimizer.param_groups[0]["lr"] = frac * self.lr
        
    def get_policy(self, states):
        states_array = np.zeros((len(states),2))
        for (row, col), state in states.items():
            states_array[state] = np.array([row, col])
        states_array = torch.as_tensor(states_array, dtype=torch.float32).to(self.device)
        with torch.no_grad():         
            policy = self.actor.get_distribution(states_array).probs.detach().cpu().numpy()
        return policy
    
    def save(self, save_dir, step):
        torch.save(self.critic.state_dict(), f"{save_dir}/networks/critic{step}.pth")
        torch.save(self.actor.state_dict(), f"{save_dir}/networks/actor{step}.pth")

    def load(self, save_dir, step):
        self.critic.load_state_dict(torch.load(f"{save_dir}/networks/critic{step}.pth"))
        self.actor.load_state_dict(torch.load(f"{save_dir}/networks/actor{step}.pth"))