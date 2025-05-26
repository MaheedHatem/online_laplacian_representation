import numpy as np
import collections
from typing import List


# H: horizon, number of transitions.
# h: 1,...,H.
# r: episodic return.
EpisodicStep = collections.namedtuple('EpisodicStep', 'step, h, H')
TransitionStep = collections.namedtuple('TransitionStep', 'state, action, terminated, truncated, next_state, reward')




class EpisodicReplayBuffer:
    """Only store full episodes.
    
    Sampling returns EpisodicStep objects.
    """

    def __init__(self, obs_dim, act_count, max_episodes: int, max_steps:int ,episode_max_steps:int, discount: float, rng: np.random.Generator, obs_mode: str, continuous: bool):
        self._max_episodes = max_episodes
        self._episode_max_steps = episode_max_steps
        self._current_size = 0
        self._steps_count = 0
        self._steps_index = 0
        self._next_idx = 0
        self._max_steps = max_steps
        self._episode_buffer = []
        self.continuous = continuous
        if isinstance(obs_dim, int):
            obs_dim = (obs_dim,)
        self._obs_dim = obs_dim
        self._obs = np.zeros((max_steps, *obs_dim), dtype=np.float32)
        if self.continuous:
            self._act = np.zeros((max_steps, act_count), dtype=np.float32)    
        else:
            self._act = np.zeros((max_steps,), dtype=np.int64)
        self._ret = np.zeros((max_steps,), dtype=np.float32)
        self._adv = np.zeros((max_steps,), dtype=np.float32)
        self._truncated = np.zeros((max_steps,), dtype=np.int64)
        self._terminated = np.zeros((max_steps,), dtype=np.int64)
        self._reward = np.zeros((max_steps,), dtype=np.int64)
        self._other_reward = np.zeros((max_steps,), dtype=np.int64)
        self._next_obs = np.zeros((max_steps, *obs_dim), dtype=np.float32)
        self._encoder_buffer = np.zeros((max_episodes, episode_max_steps, *obs_dim))
        self._episode = np.zeros((episode_max_steps, *obs_dim))
        self._current_step_in_episode = 0
        self._current_episode = 0
        self.discount = discount
        self._episodes = self.episodes = []
        self.obs_mode = obs_mode
        self.rng = rng

    @property
    def current_size(self):
        return self._current_size

    @property
    def max_episodes(self):
        return self._max_episodes

    @property
    def max_steps(self):
        return self._max_steps

    @property
    def steps_count(self):
        return self._steps_count

    def get_episodes(self,):
        return self.episodes


    def discounted_sampling(self, ranges, discount):
        """Draw samples from the discounted distribution over 0, ...., n - 1, 
        where n is a range. The input ranges is a batch of such n`s.

        The discounted distribution is defined as
        p(y = i) = (1 - discount) * discount^i / (1 - discount^n).

        This function implement inverse sampling. We first draw
        seeds from uniform[0, 1) then pass them through the inverse cdf
        floor[ log(1 - (1 - discount^n) * seeds) / log(discount) ]
        to get the samples.
        """
        assert np.min(ranges) >= 1
        assert discount >= 0 and discount <= 1
        seeds = self.rng.uniform(size=ranges.shape)
        if discount == 0:
            samples = np.zeros_like(seeds, dtype=np.int64)
        elif discount == 1:
            samples = np.floor(seeds * ranges).astype(np.int64)
        else:
            samples = (np.log(1 - (1 - np.power(discount, ranges)) * seeds) 
                    / np.log(discount))
            samples = np.floor(samples).astype(np.int64)
        return samples


    def uniform_sampling(self, ranges):
        return self.discounted_sampling(ranges, discount=1.0)

    def get_state(self, state_dict):
        if self.continuous:
            state = np.concatenate([state_dict["observation"], state_dict["desired_goal"]])
        elif self.obs_mode in ["pixels", "both"]:
            state = state_dict['pixels']
        elif self.obs_mode in ["grid", "both-grid"]:
            state = state_dict['grid']
        elif self.obs_mode in ["xy"]:
            state = state_dict['xy_agent']
        else:
            raise ValueError(f'Invalid observation mode: {self.obs_mode}')
        return state

    def add_steps(self, steps : List[TransitionStep]):
        """
        steps: a list of TransitionStep.
        """
        for step in steps:
            self._obs[self._steps_index] = self.get_state(step.state)
            self._act[self._steps_index] = step.action
            self._next_obs[self._steps_index] = self.get_state(step.next_state)
            self._reward[self._steps_index] = step.reward
            self._terminated[self._steps_index] = step.terminated
            self._truncated[self._steps_index] = step.truncated
            self._episode[self._current_step_in_episode] = self.get_state(step.state)
            self._current_step_in_episode = (self._current_step_in_episode + 1) % self._episode_max_steps
            if self._steps_count < self._max_steps:
                self._steps_count += 1
            self._steps_index += 1
            self._steps_index %= self._max_steps
            self._episode_buffer.append(step)
            # Push each step into the episode buffer until an end-of-episode
            # step is found. 
            if step.terminated or step.truncated:
                self._current_step_in_episode = 0
                self._encoder_buffer[self._current_episode, :] = np.copy(self._episode)
                self._current_episode = (self._current_episode + 1) % self._max_episodes
                # construct a formal episode
                episode = []
                H = len(self._episode_buffer)
                for h in range(H):
                    epi_step = EpisodicStep(self._episode_buffer[h], 
                            h + 1, H)
                    episode.append(epi_step)
                # save as data
                if self._next_idx == self._current_size:
                    if self._current_size < self._max_episodes:
                        self._episodes.append(episode)
                        self._current_size += 1
                        self._next_idx += 1
                    else:
                        self._episodes[0] = episode
                        self._next_idx = 1
                else:
                    self._episodes[self._next_idx] = episode
                    self._next_idx += 1
                # refresh episode buffer
                self._episode_buffer = []
                self._r = 0.0

    def sample_steps(self, batch_size):
        episode_indices = self._sample_episodes(batch_size)
        step_ranges = self._gather_episode_lengths(episode_indices)
        step_indices = self.uniform_sampling(step_ranges)
        return self._encoder_buffer[episode_indices, step_indices]

    def sample_pairs(self, batch_size, discount=0.0):
        episode_indices = self._sample_episodes(batch_size)
        step_ranges = self._gather_episode_lengths(episode_indices)
        step1_indices = self.uniform_sampling(step_ranges - 1)
        intervals = self.discounted_sampling(
            step_ranges - step1_indices - 1, discount=discount) + 1
        step2_indices = step1_indices + intervals
        return self._encoder_buffer[episode_indices, step1_indices], self._encoder_buffer[episode_indices, step2_indices]

    def _sample_episodes(self, batch_size):
        return self.rng.integers(self._current_size, size=batch_size)

    def _gather_episode_lengths(self, episode_indices):
        lengths = []
        for index in episode_indices:
            lengths.append(len(self._episodes[index]))
        return np.array(lengths, dtype=np.int64)
    
    def get_visitation_counts(self):
        """Return the visitation counts of each state."""

        visitation_counts = collections.defaultdict(int)
        for episode in self._episodes:
            for step in episode:
                agent_state = step.step.agent_state['xy_agent'].tolist()   # This assumes that the agent state is available and that it is a numpy array
                x = round(agent_state[1], 5)
                y = round(agent_state[0], 5)
                visitation_counts[(y,x)] += 1
        return visitation_counts
    
    def get_batch(self, batch_size):
        assert self._steps_count >= batch_size
        index = self.rng.choice(self._steps_count, batch_size, replace=False)
        batch = self._obs[index], self._act[index], self._next_obs[index], self._reward[index], self._terminated[index]
        return batch
    
    def get_data(self, use_gae, gae_lambda):
        assert self._steps_count == self._max_steps and self._steps_index == 0
        self._steps_count = 0
        self._steps_index = 0
        self.compute_returns(use_gae, gae_lambda)
        data = (self._obs, self._act, self._reward,
                self._ret, self._adv)
        return data
    
    def set_other_rewards(self, reward):
        self._other_reward[:] = reward[:]
    
    def get_obs_and_goal(self):
        if self.continuous:
            return self._obs[:, :2], self._obs[:, 4:]
        else:
            return self._obs, self._goal_state
    
    def compute_returns(self, use_gae, gae_lambda):

        curr_val = np.squeeze(self.get_value(self._obs))
        next_val = np.squeeze(self.get_value(self._next_obs))
        next_return = 0
        last_gae_lam = 0
        if use_gae:
            for i in reversed(range(self._max_steps)):
                delta = self._reward[i] + self._other_reward[i] + self._gamma  * next_val[i] * (1 - self._terminated[i]) - curr_val[i]
                self._adv[i] = last_gae_lam = delta + self._gamma * gae_lambda * (1 - self._terminated[i]) * last_gae_lam
            self._ret[:] = self._adv + curr_val
        else:
            for i in reversed(range(self._max_steps)):
                next_return = (1-self._truncated[i])*(self._reward[i] + self._other_reward[i] + self._gamma * (1- self._terminated[i]) * next_return) + \
                    self._truncated[i] * (self._reward[i] + self._other_reward[i] + self._gamma * next_val[i])
                self._ret[i] = next_return
            self._adv[:] = self._ret - curr_val
            
    def set_goal(self, goal_state):
        self._goal_state = np.zeros((self._max_steps, *self._obs_dim), dtype=np.float32)
        self._goal_state[:] = goal_state

    def plot_visitation_counts(self, states, env_name, grid):
        """Plot the visitation counts of each state."""

        import os
        import matplotlib.pyplot as plt
        from scipy.interpolate import Rbf

        # Get visitation counts
        visitation_counts = self.get_visitation_counts()

        # Obtain x, y, z coordinates, where z is the visitation count
        y = states[:,0]
        x = states[:,1]
        z = np.zeros_like(x)
        for i in range(states.shape[0]):
            agent_state = states[i].tolist()
            x_coord = round(agent_state[1], 5)
            y_coord = round(agent_state[0], 5)
            if (y_coord,x_coord) not in visitation_counts:
                visitation_counts[(y_coord,x_coord)] = 0
            z[i] = visitation_counts[(y_coord,x_coord)]
            
        # Calculate tile size
        x_num_tiles = np.unique(x).shape[0]
        x_tile_size = (np.max(x) - np.min(x)) / x_num_tiles
        y_num_tiles = np.unique(y).shape[0]
        y_tile_size = (np.max(y) - np.min(y)) / y_num_tiles

        # Create grid for interpolation
        ti_x = np.linspace(x.min()-x_tile_size, x.max()+x_tile_size, x_num_tiles+2)
        ti_y = np.linspace(y.min()-y_tile_size, y.max()+y_tile_size, y_num_tiles+2)
        XI, YI = np.meshgrid(ti_x, ti_y)

        # Interpolate
        rbf = Rbf(x, y, z, function='cubic')
        ZI = rbf(XI, YI)
        ZI_bounds = 85 * np.ma.masked_where(grid, np.ones_like(ZI))
        ZI_free = np.ma.masked_where(~grid, ZI)
        vmin = np.min(z)
        vmax = np.max(z)

        # Generate color mesh
        fig, ax = plt.subplots(1,1, figsize=(10,10))
        mesh = ax.pcolormesh(XI, YI, ZI_free, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.pcolormesh(XI, YI, ZI_bounds, shading='auto', cmap='Greys', vmin=0, vmax=255)
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Visitation Counts')
        plt.colorbar(mesh, ax=ax, shrink=0.5, pad=0.05)

        # Save figure
        fig_path = f'./results/visuals/{env_name}/visitation_counts.pdf'

        if not os.path.exists(os.path.dirname(fig_path)):
            os.makedirs(os.path.dirname(fig_path))

        plt.savefig(
            fig_path, 
            bbox_inches='tight', 
            dpi=300, 
            transparent=True, 
        )

        freq_visitation = z / np.sum(z)
        entropy = -np.sum(freq_visitation * np.log(freq_visitation+1e-8))
        max_entropy = -np.log(1/len(freq_visitation))
        return vmin, vmax, entropy, max_entropy, freq_visitation
        