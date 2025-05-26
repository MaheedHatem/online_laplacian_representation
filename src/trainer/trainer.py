from abc import ABC, abstractmethod
import torch.nn as nn
from src.agents import Agent
import logging
import collections
from src.utils.timer import Timer
from src.utils.summary import get_summary_str
import numpy as np
import torch
from collections.abc import Callable
from typing import Tuple
from src.env.grid.utils import load_eig
import gymnasium as gym
from src.env.wrapper.norm_obs import NormObs
from collections import OrderedDict
import gymnasium_robotics
from gymnasium.wrappers import RecordVideo
from copy import deepcopy


from matplotlib import pyplot as plt

MC_sample = collections.namedtuple(
    "MC_sample", 
    "state future_state uncorrelated_state_1 uncorrelated_state_2"
)

class Trainer(ABC):
    def __init__(self, Encoder: Callable, Agent: Callable, **kwargs ):
        super().__init__()
        gym.register_envs(gymnasium_robotics)
        self.__dict__.update((k, v) for k, v in kwargs.items())
        self.train_info = OrderedDict()
        if self.env_family == "mujoco_maze":
            self.continuous_env = True
            self.build_mujoco_maze_environment()
            self.encoder = Encoder(self.d, self.env.observation_space["achieved_goal"].shape[0], **self.encoder_params).to(self.device)
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = self.lr)
            self.agent = Agent(self.d, self.env.observation_space["observation"].shape[0] + self.env.observation_space["desired_goal"].shape[0], self.env.action_space.shape[0], self.replay_max_episodes, self.replay_max_steps, self.max_episode_steps , self.discount, self.rng, self.obs_mode, self.continuous_env, **self.agent_params)
        else:
            self.continuous_env = False
            self.build_tabular_environment()
            self.encoder = Encoder(self.d, (self.env.unwrapped.width, self.env.unwrapped.height), **self.encoder_params).to(self.device)
            self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr = self.lr)
            self.agent = Agent(self.d, self._get_obs_dim(), self.env.action_space.n, self.replay_max_episodes, self.replay_max_steps, self.max_episode_steps , self.discount, self.rng, self.obs_mode, False, **self.agent_params)
            self.agent.replay_buffer.set_goal(self.env.get_target_location())
        self.encoder_copy = deepcopy(self.encoder).to(self.device)
        self.agent.set_encoder(self.encoder_copy)
    
    def _get_obs_dim(self):
        obs_space_dict = self.env.observation_space
        if self.obs_mode in ["pixels", "both"]:
            obs_space = obs_space_dict['pixels']
        elif self.obs_mode in ["grid", "both-grid"]:
            obs_space = obs_space_dict['grid']
        elif self.obs_mode in ["xy"]:
            obs_space = obs_space_dict['xy_agent']
        else:
            raise ValueError(f'Invalid observation mode: {self.obs_mode}')
        return obs_space.shape    
    
    def build_mujoco_maze_environment(self) -> None:

        maze = [[1, 1, 1, 1, 1],
        [1, 'r', 0, 0, 1],
        [1, 1, 1, 0, 1],
        [1, 'g', 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]]
        env = gym.make(
            self.env_name, 
            render_mode="rgb_array", 
            max_episode_steps=self.max_episode_steps,
            maze_map=maze,
            continuing_task=False,
        )
        env = gym.wrappers.ClipAction(env)

        # Set environment as attribute
        self.env = env
        self.env.reset(seed=self.seed)

        eval_env = gym.make(
            self.env_name, 
            render_mode="rgb_array", 
            max_episode_steps=self.max_episode_steps,
            maze_map=maze,
            continuing_task=False,
        )


        # Set environment as attribute
        #eval_env = RecordVideo(gym.wrappers.ClipAction(eval_env), video_folder="./videos", episode_trigger=lambda x: True)
        eval_env = gym.wrappers.ClipAction(eval_env)
        self.eval_env = eval_env
        self.eval_env.reset(seed=self.seed+1)

        # Save eigenvectors and eigenvalues
        # if eig_not_found and self.save_eig:
        #     self.env.save_eigenpairs(path_eig)

    def build_tabular_environment(self) -> None:
        # Load eigenvectors and eigenvalues of the transition dynamics matrix (if they exist)
        # path_eig = f'./src/env/grid/eigval/{self.env_name}.npz'
        # eig, eig_not_found = load_eig(path_eig)

        # Create environment
        path_txt_grid = f'./src/env/grid/txts/{self.env_name}.txt'
        env = gym.make(
            self.env_family, 
            path=path_txt_grid, 
            render_mode="rgb_array", 
            use_target=True, 
            obs_mode=self.obs_mode,
            window_size=self.window_size,
            max_episode_steps=self.max_episode_steps,
            max_steps=self.max_episode_steps
        )

        # Wrap environment with observation normalization
        if self.encoder_type != "matrix":
            obs_wrapper = lambda e: NormObs(
            e, reduction_factor=self.reduction_factor)
            env = obs_wrapper(env)

        # Set environment as attribute
        self.env = env
        self.env.reset(seed=self.seed)

        eval_env = gym.make(
            self.env_family, 
            path=path_txt_grid, 
            render_mode="rgb_array", 
            use_target=False, 
            obs_mode=self.obs_mode,
            window_size=self.window_size,
            max_episode_steps=self.max_episode_steps,
            goal_state=self.env.unwrapped.get_goal_state(),
            max_steps=self.max_episode_steps
        )

        # Wrap environment with observation normalization
        if self.encoder_type != "matrix":
            obs_wrapper = lambda e: NormObs(
            e, reduction_factor=self.reduction_factor)
            eval_env = obs_wrapper(eval_env)

        # Set environment as attribute
        self.eval_env = eval_env
        self.eval_env.reset(seed=self.seed+1)

        # Save eigenvectors and eigenvalues
        # if eig_not_found and self.save_eig:
        #     self.env.save_eigenpairs(path_eig)

    def load_eigensystem(self, policy=None):
        # Log environment eigenvalues
        self.env.unwrapped.recompute_eigensystem(policy, self.use_stationary_for_similarity)
        self.env.unwrapped.round_eigenvalues(self.eigval_precision_order)
        eigenvalues = self.env.unwrapped.get_eigenvalues()
        #logging.info(f'Environment: {self.env_name}')
        #logging.info(f'Environment eigenvalues: {eigenvalues}')

        # Create eigenvector dictionary
        real_eigval = eigenvalues[:self.d]
        real_eigvec = self.env.unwrapped.get_eigenvectors()[:,:self.d]

        assert not np.isnan(real_eigvec).any(), \
            f'NaN values in the real eigenvectors: {real_eigvec}'


        #real_norms = np.linalg.norm(real_eigvec, axis=0, keepdims=True)
        real_eigvec_norm = real_eigvec

        # Check if any NaN values are present
        assert not np.isnan(real_eigvec_norm).any(), \
            f'NaN values in the real eigenvectors: {real_eigvec_norm}'

        # Store eigenvectors in a dictionary corresponding to each eigenvalue
        eigvec_dict = {}
        for i, eigval in enumerate(real_eigval):
            if eigval not in eigvec_dict:
                eigvec_dict[eigval] = []
            eigvec_dict[eigval].append(real_eigvec_norm[:,i])
        self.eigvec_dict = eigvec_dict
        
        # Print multiplicity of first eigenvalues
        multiplicities = [len(eigvec_dict[eigval]) for eigval in eigvec_dict.keys()]
        #for i, eigval in enumerate(eigvec_dict.keys()):
        #    logging.info(f'Eigenvalue {eigval} has multiplicity {multiplicities[i]}')
        if self.env.unwrapped.mixing_time > self.max_episode_steps:
            logging.warning(f"Mixing time is {self.env.unwrapped.mixing_time}")

    def _get_train_batch(self):
        state, future_state = self.agent.replay_buffer.sample_pairs(
                batch_size=self.batch_size,
                discount=self.discount,
                )
        uncorrelated_state_1 = self.agent.replay_buffer.sample_steps(self.batch_size)
        uncorrelated_state_2 = self.agent.replay_buffer.sample_steps(self.batch_size)
        if self.continuous_env:
            state = state[:, :2]
            future_state = future_state[:, :2]
            uncorrelated_state_1 = uncorrelated_state_1[:, :2]
            uncorrelated_state_2 = uncorrelated_state_2[:, :2]
        #state, future_state, uncorrelated_state_1, uncorrelated_state_2 = map(
        #    self._get_obs_batch, [state, future_state, uncorrelated_state_1, uncorrelated_state_2])
        batch = MC_sample(state, future_state, uncorrelated_state_1, uncorrelated_state_2)
        return batch

    def _get_obs_batch(self, steps):
        if self.obs_mode in ["xy"]:
            obs_batch = [s.step.state["xy_agent"].astype(np.float32)
                    for s in steps]
            return np.stack(obs_batch, axis=0)
        elif self.obs_mode in ["pixels", "both"]:
            obs_batch = [s.step.state["pixels"] for s in steps]
            obs_batch = np.stack(obs_batch, axis=0).astype(np.float32)/255
            return obs_batch
        elif self.obs_mode in ["grid", "both-grid"]:
            obs_batch = [s.step.state["grid"].astype(np.float32)/255 for s in steps]
            obs_batch = np.stack(obs_batch, axis=0)
            return obs_batch
    
    def train(self) -> None:
        timer = Timer()
        train = (not self.load_encoder) or (not self.load_agent)
        timer.set_step(0)
        scores = np.zeros((((self.total_train_steps+1) // self.print_freq)+1, 3))
        eigen_decomposition_timer = Timer()
        if not self.continuous_env:
            policy = self.agent.get_policy(self.env.get_states_indices())
        
            self.load_eigensystem(policy)
        encoder_updates_count = int(self.updates_per_step * self.train_every)
        for step in range(0, self.total_train_steps+1, self.train_every):
            self.agent.update_hyperparameters(step)
            if train:
                self.agent.collect_experience(self.env, self.train_every)
            is_last_step = (step) == self.total_train_steps

            is_save_step = (
                self.save_model 
                and (
                    (((step) % self.save_model_every) == 0)
                    or is_last_step
                )
            )
            if is_save_step:
                if not self.load_agent:
                    self.agent.save(self.save_dir, step)
                if not self.load_encoder:
                    self.encoder.save(self.save_dir, step)
            if (not self.load_encoder) and self.train_encoder:
                if  step >= self.train_encoder_after:
                    for _ in range(encoder_updates_count):
                        train_batch = self._get_train_batch()
                        metrics = self.train_step(train_batch)
                        self.additional_update_step()
            elif self.load_encoder and (step % self.save_model_every == 0 or is_last_step):
                self.encoder.load(self.save_dir, step)
            is_log_step = ((step) % self.print_freq) == 0
            if step % self.update_policy_encoder_freq == 0:
                with torch.no_grad():
                    for p, p_targ in zip(self.encoder.parameters(), self.encoder_copy.parameters()):
                        p_targ.data.copy_(p.data)
            if is_log_step:
                steps_per_sec = timer.steps_per_sec(step)
                eigen_decomposition_timer.reset()
                if not self.continuous_env:
                    policy = self.agent.get_policy(self.env.get_states_indices())
                    self.load_eigensystem(policy)
                logging.info(f"Eigendecomposition time {eigen_decomposition_timer.time_cost()}")
                timer.set_step(step)
            if not self.load_agent:
                if  step > self.train_after:
                    self.agent.train()   
            elif step % self.save_model_every == 0 or is_last_step:
                self.agent.load(self.save_dir, step)   
            # Save and print info
            if is_log_step:
                if  (not self.load_encoder) and self.train_encoder and step >= self.train_encoder_after:
                    losses = metrics[:-1]
                    metrics_dict = metrics[-1]
                    self.train_info['loss_total'] = float(losses[0].cpu().detach().numpy())
                    self.train_info['graph_loss'] = float(losses[1].cpu().detach().numpy())
                    self.train_info['dual_loss'] = float(losses[2].cpu().detach().numpy())
                    self.train_info['barrier_loss'] = float(losses[3].cpu().detach().numpy())
                else:
                    metrics_dict = {}
                
                self.train_info['average_reward'] = self.agent.eval(self.eval_env, self.eval_episodes)
                if not self.continuous_env:
                    metrics_dict = self._compute_additional_metrics(metrics_dict)
                
                metrics_dict['grad_step'] = step
                metrics_dict['examples'] = step * self.batch_size                
                metrics_dict['wall_clock_time'] = timer.time_cost()
                
                logging.info(f'Training steps per second: {steps_per_sec:.4g}.')

                self._print_train_info(step)
                scores[step//self.print_freq, 0] = step
                if not self.continuous_env:
                    scores[step//self.print_freq, 1] = self.train_info['cos_sim']
                scores[step//self.print_freq, 2] = self.train_info['average_reward']
                    
        time_cost = timer.time_cost()
        logging.info(f'Training finished, time cost {time_cost:.4g}s.')
        np.savetxt(f"{self.save_dir}/results.csv", scores, delimiter=',')
        if not self.continuous_env:
            states = self.get_states()
            # Get approximated eigenvectors
            approx_eigvec = self.encoder(states).cpu().detach().numpy()


            # Normalize approximated eigenvectors
            measure = self.env.unwrapped.get_stat_distribution()
            repeated_stat_distr = np.broadcast_to(np.expand_dims(measure, axis=1), approx_eigvec.shape)
            norms = np.sqrt((approx_eigvec * approx_eigvec * repeated_stat_distr).sum(axis = 0, keepdims=True))
            approx_eigvec = approx_eigvec / norms.clip(min=1e-10)
            
            logging.info("angles")
            logging.info(np.round(np.arccos((np.round(approx_eigvec.T @ (approx_eigvec * repeated_stat_distr), 2)))*180/np.pi))
            logging.info("norms")
            logging.info(norms)
            
            unique_real_eigval = sorted(self.eigvec_dict.keys(), reverse=True)
            logging.info("True eigval")
            logging.info(unique_real_eigval)


    def _print_train_info(self, step):
        summary_str = get_summary_str(
                step=step, info=self.train_info)
        logging.info(summary_str)   

    def encode_states(
            self,
            train_batch: MC_sample,
        ) -> Tuple[torch.Tensor]:

        # Compute start representations
        obs_type = torch.float32
        if self.encoder_type == "matrix":
            obs_type = torch.long
        start_representation = self.encoder(torch.as_tensor(train_batch.state, dtype=obs_type).to(self.device))
        constraint_representation_1 = self.encoder(torch.as_tensor(train_batch.uncorrelated_state_1, dtype=obs_type).to(self.device))

        # Compute end representations
        end_representation = self.encoder(torch.as_tensor(train_batch.future_state, dtype=obs_type).to(self.device))
        constraint_representation_2 = self.encoder(torch.as_tensor(train_batch.uncorrelated_state_2, dtype=obs_type).to(self.device))

        return (
            start_representation, end_representation, 
            constraint_representation_1, 
            constraint_representation_2,
        )

    def get_states(self):
        state_dict = self.env.get_states()
        if self.obs_mode in ["pixels", "both"]:
            states = state_dict['pixels']
        elif self.obs_mode in ["grid", "both-grid"]:
            states = state_dict['grid']
        elif self.obs_mode in ["xy"]:
            states = state_dict['xy_agent']
        else:
            raise ValueError(f'Invalid observation mode: {self.obs_mode}')
        obs_type = torch.float32
        if self.encoder_type == "matrix":
            obs_type = torch.long
        return torch.as_tensor(states, dtype=obs_type).to(self.device)

    def _compute_additional_metrics(self, metrics_dict):

        # Compute cosine similarities
        cs, sim = self.compute_cosine_similarity(batch_size=self.batch_size)
        maximal_cs, maximal_sim = self.compute_maximal_cosine_similarity()
        
        # Store metrics
        metrics_dict['cosine_similarity'] = cs
        metrics_dict['maximal_cosine_similarity'] = maximal_cs

        for feature in range(len(sim)):   # Similarities for each feature
            metrics_dict[f'cosine_similarity_{feature}'] = sim[feature]
            metrics_dict[f'maximal_cosine_similarity_{feature}'] = maximal_sim[feature]

        # Log in train_info to print
        self.train_info['cos_sim'] = cs
        self.train_info['max_cos_sim'] = maximal_cs
        
        return metrics_dict

    def compute_cosine_similarity(self, batch_size=32):
        # Get states
        states = self.get_states()             

        # Get approximated eigenvectors
        approx_eigvec = self.encoder(states).cpu().detach().numpy()

        measure = self.env.unwrapped.get_stat_distribution()
        # Normalize approximated eigenvectors
        if self.use_stationary_for_similarity:
            repeated_stat_distr = np.broadcast_to(np.expand_dims(measure, axis=1), approx_eigvec.shape)
            norms = np.sqrt((approx_eigvec * approx_eigvec * repeated_stat_distr).sum(axis = 0, keepdims=True))
        else:
            norms = np.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms.clip(min=1e-10)
        
        
        # Compute cosine similarities for both directions
        unique_real_eigval = sorted(self.eigvec_dict.keys(), reverse=True)

        id_ = 0
        similarities = []
        for i, eigval in enumerate(unique_real_eigval):
            multiplicity = len(self.eigvec_dict[eigval])
            
            # Compute cosine similarity
            if multiplicity == 1:
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = self.eigvec_dict[eigval][0]
                current_approx_eigvec = approx_eigvec[:,id_]

                # Check if any NaN values are present
                assert not np.isnan(current_approx_eigvec).any(), \
                    f'NaN values in the approximated eigenvector: {current_approx_eigvec}'
                
                assert not np.isnan(current_real_eigvec).any(), \
                    f'NaN values in the real eigenvector: {current_real_eigvec}'

                # Compute cosine similarity
                if self.use_stationary_for_similarity:
                    pos_sim = (current_real_eigvec*current_approx_eigvec*measure).sum()
                else:
                    pos_sim = (current_real_eigvec).dot(current_approx_eigvec)
                similarities.append(np.maximum(pos_sim, -pos_sim))

            else: #TODO: Add implementation of similarity with respect to stationary dist
                logging.warn("multiplicity is not 1")
                # Get eigenvectors associated with the current eigenvalue
                current_real_eigvec = self.eigvec_dict[eigval]
                current_approx_eigvec = approx_eigvec[:,id_:id_+multiplicity]
                
                # Rotate approximated eigenvectors to match the space spanned by the real eigenvectors
                optimal_approx_eigvec = self.rotate_eigenvectors(
                    current_real_eigvec, current_approx_eigvec)

                norms = np.linalg.norm(optimal_approx_eigvec, axis=0, keepdims=True)
                optimal_approx_eigvec = optimal_approx_eigvec / norms.clip(min=1e-10)   # We normalize, since the cosine similarity is invariant to scaling
                
                # Compute cosine similarity
                for j in range(multiplicity):
                    real = current_real_eigvec[j]
                    approx = optimal_approx_eigvec[:,j]
                    pos_sim = (real).dot(approx)
                    similarities.append(np.maximum(pos_sim, -pos_sim))

            id_ += multiplicity

        # Convert to array
        similarities = np.array(similarities)

        # Compute average cosine similarity
        cosine_similarity = similarities.mean()

        assert not np.isnan(similarities).any(), \
            f'NaN values in the cosine similarities: {similarities}'

        return cosine_similarity, similarities

    def compute_maximal_cosine_similarity(self):
        #TODO: Add similarity with respect to stationary distribution
        # Get states
        states = self.get_states()

        # Get approximated eigenvectors
        approx_eigvec = self.encoder(states).cpu().detach().numpy()
        norms = np.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms.clip(min=1e-10)
        
        # Select rotation function
        rotation_function = self.rotate_eigenvectors
        
        real_eigvec = []
        for eigval in self.eigvec_dict.keys():
            real_eigvec = real_eigvec + self.eigvec_dict[eigval]
                
        # Rotate approximated eigenvectors to match the space spanned by the real eigenvectors
        optimal_approx_eigvec = rotation_function(
            real_eigvec, approx_eigvec)
        norms = np.linalg.norm(optimal_approx_eigvec, axis=0, keepdims=True)
        optimal_approx_eigvec = optimal_approx_eigvec / norms.clip(min=1e-10)   # We normalize, since the cosine similarity is invariant to scaling
        
        # Compute cosine similarity
        similarities = []
        for j in range(self.d):
            real = real_eigvec[j]
            approx = optimal_approx_eigvec[:,j]
            pos_sim = (real).dot(approx)
            similarities.append(np.maximum(pos_sim, -pos_sim))

        # Convert to array
        similarities = np.array(similarities)

        # Compute average cosine similarity
        cosine_similarity = similarities.mean()

        return cosine_similarity, similarities

    def rotate_eigenvectors(
            self, 
            u_list: list, 
            E: np.ndarray
        ) -> np.ndarray:
        '''
            Rotate the eigenvectors in E to match the 
            eigenvectors in u_list as close as possible.
            That is, we are finding the optimal basis of
            the subspace spanned by the eigenvectors in E
            such that the angle between the eigenvectors
            in u_list and the rotated eigenvectors is
            minimized.
        '''
        rotation_vectors = []
        # Compute first eigenvector
        u1 = u_list[0].reshape(-1,1)
        w1_times_lambda_1 = 0.5*E.T.dot(u1)
        w1 = w1_times_lambda_1 / np.linalg.norm(w1_times_lambda_1).clip(min=1e-10)
        rotation_vectors.append(w1)

        # Compute remaining eigenvectors
        for k in range(1, len(u_list)):
            uk = u_list[k].reshape(-1,1)
            Wk = np.concatenate(rotation_vectors, axis=1)
            improper_wk = E.T.dot(uk)
            bk = Wk.T.dot(improper_wk)
            Ak = Wk.T.dot(Wk)
            mu_k = np.linalg.solve(Ak, bk)
            wk_times_lambda_k = 0.5*(improper_wk - Wk.dot(mu_k))
            wk = wk_times_lambda_k / np.linalg.norm(wk_times_lambda_k).clip(min=1e-10)
            rotation_vectors.append(wk)

        # Use rotation vectors as columns of the optimal rotation matrix
        R = np.concatenate(rotation_vectors, axis=1)

        # Obtain list of rotated eigenvectors
        rotated_eigvec = E.dot(R)
        return rotated_eigvec
    
    def plot_env(self):
        plt.imshow(self.env.unwrapped.render())
        plt.savefig(f'test.pdf')

    @abstractmethod
    def train_step(self, train_batch: MC_sample):
        raise NotImplementedError

    @abstractmethod
    def additional_update_step(self):
        raise NotImplementedError

    @abstractmethod
    def update_training_state(self, *args, **kwargs):
        raise NotImplementedError