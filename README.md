# Online Laplacian-Based Representation Learning in Reinforcement Learning

This is the official code implementation for the paper "Online Laplacian-Based Representation Learning in Reinforcement Learning".

## Setup
In order to run the files install python>=3.9 then run:

`pip install -r requirements.txt `

## Training

To train, run the command:

`python main.py --save_dir <output_directory> [--seed <seed>] [--config_file <config_file>.yaml]`

Config files must be under src/hyperparam. Available configs are:
- al.yaml
- al_ppo.yaml

## Comparisons
To plot a comparison between multiple runs, create a .yaml file <filename.yaml> under Comparisons folder with the same format as other files in the directory then run the following command.

`python plot.py <comparison_file_name_without_extension> [optional_root_folder]`

Note that comparisons require that a full training run to guarantee the required files are created in the output folder.
