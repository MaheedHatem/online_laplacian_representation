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
import os

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
    os.environ["MUJOCO_GL"] = "egl"
    command_args = init_parser()
    hyperparam_file = f'./src/hyperparam/{command_args.config_file}'
    with open(hyperparam_file, 'r') as f:
        hyperparams = yaml.safe_load(f)
    
    for k, v in vars(command_args).items():
        if v is not None:
            hyperparams[k] = v

    pathlib.Path(hyperparams['save_dir']).mkdir(exist_ok=True)
    if hyperparams["load_hyper_params"]:
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
    logging.info(command_args)
    try:
        trainer.train()
    except Exception as e:
        logging.error(traceback.format_exc())
        raise e
