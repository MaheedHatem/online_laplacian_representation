from abc import ABC, abstractmethod
import collections
import gymnasium as gym
from src.agents.replay import EpisodicReplayBuffer, TransitionStep
import numpy as np

class Agent(ABC):
    def __init__(self, d:int, obs_dim: int, num_actions: int, replay_max_episodes: int, replay_max_steps:int, episode_max_steps:int , discount: float, rng: np.random.Generator, obs_mode: str, continuous_env: bool, **kwargs):
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.d = d
        self.continuous_env = continuous_env
        self.replay_buffer = EpisodicReplayBuffer(obs_dim, num_actions, replay_max_episodes, replay_max_steps, episode_max_steps , discount, rng, obs_mode, continuous_env)
        self.episode_done = True
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.rng = rng
    
    @abstractmethod
    def act(self, state, env=None, deterministic=False):
        raise NotImplementedError

    def collect_experience(self, env: gym.Env, num_steps: int):
        steps = []
        for _ in range(num_steps):
            if self.episode_done:
                # Reset environment if episode is done
                #self.state = env.reset(seed=int(self.rng.integers(2**31)))[0]
                self.state = env.reset()[0]
                self.episode_done = False
            action = self.act(self.state, env)
            if not self.continuous_env:
                action = int(action)
            # Take action and get next observation
            observation, reward, terminated, truncated, info = env.step(action)
            self.episode_done = terminated or truncated
            step = TransitionStep(self.state, action, terminated, truncated, observation, reward)
            steps.append(step)
            self.state = observation

        self.replay_buffer.add_steps(steps)

    def eval(self, env: gym.Env, num_episodes: int):
        steps = []
        episodes = 0
        episodes_rewards = 0
        state = env.reset()[0]
        episode_done = False
        while episodes < num_episodes:
            action = self.act(state, env)
            if not self.continuous_env:
                action = int(action)
            # Take action and get next observation
            observation, reward, terminated, truncated, info = env.step(action)
            episodes_rewards += reward
            episode_done = terminated or truncated
            state = observation
            if episode_done:
                state = env.reset()[0]
                episode_done = False
                episodes += 1

        return episodes_rewards / num_episodes
    

    def set_encoder(self, encoder):
        self.encoder = encoder

    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def get_policy(self, states):
        raise NotImplementedError
    
    @abstractmethod
    def update_hyperparameters(self, step):
        raise NotImplementedError
 
    @abstractmethod   
    def save(self, save_dir, step):
        raise NotImplementedError

    @abstractmethod
    def load(self, save_dir, step):
        raise NotImplementedError