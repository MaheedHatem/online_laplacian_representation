import os
import yaml
from argparse import ArgumentParser
import torch
import numpy as np
import random
from src.utils.logger import init_logging
import logging
from src.agents import *
from src.trainer import *
from src.encoder import *
import pathlib
import logging
import shutil
import traceback
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

def init_parser():
    parser = ArgumentParser()
    parser.add_argument(
        '--config_file', 
        type=str, 
        default= 'al.yaml',
        help='Configuration file to use.'
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default="./output", 
        help='Directory to save the model.'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=None, 
        help='Batch size.'
    )
    parser.add_argument(
        '--discount', 
        type=float, 
        default=None, 
        help='Lambda discount used for sampling states.'
    )
    parser.add_argument(
        '--total_train_steps', 
        type=int, 
        default=None, 
        help='Number of training steps for laplacian encoder.'
    )
    parser.add_argument(
        '--max_episode_steps', 
        type=int, 
        default=None, 
        help='Maximum trajectory length.'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=None, 
        help='Seed for random number generators.'
    )
    parser.add_argument(
        '--env_name', 
        type=str, 
        default=None, 
        help='Environment name.'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=None, 
        help='Learning rate of the Adam optimizer used to train the laplacian encoder.'
    )
    parser.add_argument(
        '--hidden_dims',
        nargs='+',
        type=int,
        help='Hidden dimensions of the laplacian encoder.'
    )
    parser.add_argument(
        '--barrier_initial_val', 
        type=float, 
        default=None, 
        help='Initial value for barrier coefficient in the quadratic penalty.'
    )
    parser.add_argument(
        '--lr_barrier_coefs', 
        type=float, 
        default=None, 
        help='Learning rate of the barrier coefficient in the quadratic penalty.'
    )
    parser.add_argument(
        '--print_freq', 
        type=int, 
        default=None, 
        help='Logging frequency.'
    )
    parser.add_argument(
        '--updates_per_step', 
        type=int, 
        default=None, 
        help='Updates per step.'
    )
    parser.add_argument(
        '--algorithm', 
        type=str, 
        default=None, 
        help='allo or gdo.'
    )
    parser.add_argument(
        '--replay_max_episodes', 
        type=int, 
        default=None, 
        help='Maximum episodes in replay.'
    )
    
    
    return parser.parse_args()

if __name__ == "__main__":
    command_args = init_parser()
    hyperparam_file = f'./src/hyperparam/{command_args.config_file}'
    with open(hyperparam_file, 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    for k, v in vars(command_args).items():
        if v is not None:
            hyperparams[k] = v

    pathlib.Path(hyperparams['save_dir']).mkdir(exist_ok=True)
    if hyperparams["load_agent"] or hyperparams["load_encoder"]:
        load_agent = hyperparams["load_agent"]
        load_encoder = hyperparams["load_encoder"]
        hyperparam_file = f"{hyperparams['save_dir']}/hyper.yaml"
        with open(hyperparam_file, 'r') as f:
            hyperparams = yaml.safe_load(f)
        
        for k, v in vars(command_args).items():
            if v is not None:
                hyperparams[k] = v
        hyperparams["load_agent"] = load_agent
        hyperparams["load_encoder"] = load_encoder
    else:
        shutil.copyfile(hyperparam_file, f"{hyperparams['save_dir']}/hyper.yaml")
    
    
    pathlib.Path(f"{hyperparams['save_dir']}/networks").mkdir(exist_ok=True)
    init_logging(f"{hyperparams['save_dir']}/log.log")
    rng = np.random.default_rng(hyperparams['seed'])
    random.seed(hyperparams['seed'])
    torch.manual_seed(hyperparams['seed'])
    hyperparams.update({"rng": rng})

    encoder_type = hyperparams["encoder_type"]
    if encoder_type == 'matrix':
        Encoder = TabularRepresentation
    elif encoder_type == 'mlp':
        Encoder = MLPEncoder
    else:
        raise ValueError(f'Encoder {encoder_type} is not supported.')

    agent_type = hyperparams["agent_type"]
    if agent_type == 'uniform':
        Agent = UniformRandomPolicy
    elif agent_type == 'dqn':
        Agent = DQN
    elif agent_type == 'ppo':
        Agent = PPO
    else:
        raise ValueError(f'Agent {agent_type} is not supported.')

    algorithm = hyperparams["algorithm"]
    if algorithm == 'allo':
        Trainer = AugmentedLaplacianLagrangianOptimizer
    elif algorithm == 'allo_matrix':
        Trainer = AugmentedLaplacianLagrangianMatrixOptimizer
    elif algorithm == 'gdo':
        Trainer = GDOOptimizer
    else:
        raise ValueError(f'Algorithm {algorithm} is not supported.')

    trainer = Trainer(Encoder, Agent, **hyperparams)
    logging.info(hyperparams)
    for step in [100000]:
    
        trainer.encoder.load(trainer.save_dir, step)
        trainer.agent.load(trainer.save_dir, step)
        trainer.agent.update_hyperparameters(step)
        
        state_indices = trainer.env.unwrapped.get_states_indices()
        policy = trainer.agent.get_policy(state_indices)
        # Set seed
        trainer.load_eigensystem(policy)
        states = trainer.get_states()             

        

        # Get approximated eigenvectors
        #approx_eigvec = trainer.encoder(states).cpu().detach().numpy()
        approx_eigvec = trainer.env.unwrapped.get_eigenvectors()[:,:trainer.d]
        measure = trainer.env.unwrapped.get_stat_distribution()

        repeated_stat_distr = np.broadcast_to(np.expand_dims(measure, axis=1), approx_eigvec.shape)
        #norms = np.sqrt((approx_eigvec * approx_eigvec * repeated_stat_distr).sum(axis = 0, keepdims=True))
        norms = np.linalg.norm(approx_eigvec, axis=0, keepdims=True)
        approx_eigvec = approx_eigvec / norms.clip(min=1e-10)


        cmap = plt.cm.Blues  # Change colormap as needed
        cmap.set_bad(color='grey')  #
        grid = trainer.env.unwrapped.grid
        grid = grid.astype(np.float32)
        goal_state = trainer.env.get_goal_state()
        grid[grid==0] = np.NaN
        goal_row = 0
        goal_col = 0
        for (row, col), state in state_indices.items():
            if goal_state == state:
                goal_row = row
                goal_col = col
            grid[row,col] = approx_eigvec[state, 1]
            #grid[row,col] = np.sqrt(((approx_eigvec[state,:] - approx_eigvec[goal_state,:])**2).sum())

        # Create the plot
        fig, ax = plt.subplots(figsize=(6, 6))
        img = ax.imshow(grid, cmap=cmap, origin='upper')
        plt.colorbar(img)
        print(grid)
        plt.xticks([])  # Remove x-axis ticks
        plt.yticks([])  # Remove y-axis ticks
        rect = patches.Rectangle((goal_col - 0.5, goal_row - 0.5), 1, 1, linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        #plt.grid(which='major', color='gray', linestyle='-', linewidth=0.5)

        # Save as a PDF
        plt.savefig(f"src/Figures/{hyperparams['env_name']}_{step}_dqn.pdf", format='pdf', bbox_inches='tight')
        plt.close()