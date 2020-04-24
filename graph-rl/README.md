# Deep Reinforcement Learning using Graph Neural Networks

## Installation

Install the package by running `pip install -e .` in the root directory.
The code requires commit 51efc31382936503fc36919cf31bb3688b56f128 from sacred, and requires openai baselines to be installed from source.

## Generating environment map

Run the `python gen_commands/generate_environments.py` folder to generate the environment maps.

## Training agents
Run these commands to train agents. Change the env_config file to use a different map.

Warehouse dqn:
`python3.7 scripts/warehouse_v1/train_dqn.py with env_configs/warehouse_v1/one_one.json model_configs/reduce.json model_configs/qhead.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 agent.min_train_episodes=40000`

Warehouse graph dqn:
`python3.7 scripts/warehouse_v1/train_dqn_graph.py with env_configs/warehouse_v1/two_one.json model_configs/dgconv_original.json model_configs/qhead.json hyper_configs/slow.json agent.opt.kwargs.lr=1e-4 agent.min_train_episodes=40000 arch.trunk.use_orth_init=True`

Warehouse a2c command:
`python3.7 scripts/warehouse_v1/train_ac.py with env_configs/warehouse_v1/one_one.json model_configs/reduce.json model_configs/vcathead.json hyper_configs/slow_ac.json hyper_configs/movable.json agent.min_train_episodes=40000`

Warehouse graph a2c:
`python3.7 scripts/warehouse_v1/train_ac_graph.py with env_configs/warehouse_v1/one_one.json model_configs/dgconv_original.json model_configs/vcathead.json hyper_configs/slow_ac.json hyper_configs/movable.json agent.min_train_episodes=40000 arch.trunk.use_orth_init=True`

Pacman dqn:
`python3.7 scripts/pacman_v1/train_dqn.py with env_configs/pacman_v1/mediumClassic_random.json model_configs/reduce_large.json model_configs/qhead.json hyper_configs/pacman_dqn_hypers.json`

Pacman graph dqn:
`python3.7 scripts/pacman_v1/train_dqn_graph.py with env_configs/pacman_v1/mediumClassic_random.json model_configs/dgconv_large.json model_configs/qhead.json hyper_configs/pacman_dqn_hypers.json arch.trunk.use_orth_init=True`

Pacman a2c:
`ppython3.7 scripts/pacman_v1/train_ac.py with env_configs/pacman_v1/mediumClassic_random.json model_configs/reduce_large.json model_configs/vcathead.json hyper_configs/pacman_ac_hypers.json`

Pacman graph a2c:
`python3.7 scripts/pacman_v1/train_ac_graph.py with env_configs/pacman_v1/mediumClassic_random.json model_configs/dgconv_large.json model_configs/vcathead.json hyper_configs/pacman_ac_hypers.json arch.trunk.use_orth_init=True`


