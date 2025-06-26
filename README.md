# Linguistic Prior Reinforcement Learning

A research project exploring the integration of linguistic priors and knowledge graphs with reinforcement learning agents in structured environments, particularly focusing on Minecraft/Malmo environments.

## Overview

This project investigates how linguistic knowledge and graph-based representations can enhance reinforcement learning agents' performance in complex environments. By incorporating Graph Convolutional Networks (GCNs) and knowledge graphs, the agents can leverage structured semantic information to improve decision-making and generalization.

## Project Structure

```
ling_prior_rl/
├── gcn.py                      # Graph Convolutional Network implementation
├── gcn_raw.py                  # Alternative GCN implementation
├── graph-rl/                   # Main RL framework with graph integration
│   ├── baselines/              # Baseline RL algorithms
│   ├── graphrl/                # Graph-enhanced RL implementations
│   ├── environments/           # Environment definitions
│   ├── model_configs/          # Model configuration files
│   ├── hyper_configs/          # Hyperparameter configurations
│   └── scripts/                # Training and evaluation scripts
├── knowledge_graphs/           # Knowledge graph resources and utilities
├── minecraft/                  # Minecraft/Malmo environment wrappers
│   ├── malmo_env_env.py       # Base Malmo environment
│   ├── malmo_specialized_env.py # Specialized environment variants
│   └── dqn_experiments/       # DQN experiment configurations
└── wordnet/                   # WordNet integration for linguistic priors
```

## Key Components

### 1. Graph Convolutional Networks (GCN)
- **gcn.py**: Implements a GCN module that embeds game states using graph-structured knowledge
- Supports node embeddings and object type embeddings
- Integrates graph embeddings into state representations

### 2. Reinforcement Learning Framework
- Based on the [Intel AI Graph-RL](https://github.com/IntelAI/graph-rl) framework
- Supports DQN and other baseline algorithms
- Enhanced with graph-based architectures for improved performance

### 3. Minecraft/Malmo Integration
- Custom Malmo environment wrappers for structured tasks
- Support for various goal-oriented scenarios
- Server-based architecture for distributed training

### 4. Knowledge Graphs
- Integration of semantic knowledge through graph structures
- Support for task-specific knowledge graphs (e.g., pickaxe_skyline.json)
- Enables reasoning about object relationships and affordances

## Installation

### Prerequisites
- Python 3.7
- CUDA 9.2 (for GPU support)
- Conda/Miniconda

### Setup Instructions

1. **Create conda environment:**
   ```bash
   conda create -n graph-rl python=3.7
   conda activate graph-rl
   ```

2. **Clone and setup the graph-rl framework:**
   ```bash
   git clone https://github.com/IntelAI/graph-rl.git
   cd graph-rl
   
   # Install Sacred
   git clone https://github.com/IDSIA/sacred.git
   cd sacred
   git checkout 51efc31382936503fc36919cf31bb3688b56f128
   python setup.py install
   cd ..
   
   # Install Baselines
   git clone https://github.com/openai/baselines.git
   cd baselines
   pip install tensorflow-gpu==1.14
   pip install cloudpickle~=1.2.0
   pip install -e .
   cd ..
   
   # Install graph-rl
   pip install -e .
   ```

3. **Install additional dependencies:**
   ```bash
   pip install torch visdom_observer torchvision pycolab torch_geometric
   pip install torch-scatter==latest+cu92 torch-sparse==latest+cu92 -f https://pytorch-geometric.com/whl/torch-1.4.0.html
   ```

4. **Setup Malmo (for Minecraft environments):**
   ```bash
   pip install malmoenv
   python -c "import malmoenv.bootstrap; malmoenv.bootstrap.download()"
   ```

5. **Generate environments:**
   ```bash
   python gen_commands/generate_environments.py
   ```

## Usage

### Running Baseline Experiments

Train a DQN baseline agent:
```bash
python scripts/warehouse_v1/train_dqn.py with \
    env_configs/warehouse_v1/one_one.json \
    model_configs/reduce.json \
    model_configs/qhead.json \
    hyper_configs/slow.json \
    agent.opt.kwargs.lr=1e-4 \
    agent.min_train_episodes=40000
```

### Running Graph-Enhanced Experiments

Train a graph-enhanced DQN agent:
```bash
python scripts/malmo_v0/train_dqn_graph.py with \
    env.train.kg.file=pickaxe_skyline.json \
    env.test.kg.file=pickaxe_skyline.json \
    model_configs/dgconv_large.json \
    model_configs/qhead.json \
    hyper_configs/slow.json \
    agent.opt.kwargs.lr=1e-4 \
    agent.min_train_episodes=40000 \
    arch.trunk.use_orth_init=True \
    agent.experiment_name='pickaxe_stone' \
    port=9000 \
    address='127.0.0.1'
```

### Running Malmo Server

To run experiments with Minecraft/Malmo environments:

1. Start a VNC session (if on a cluster)
2. Navigate to the project directory and activate the environment
3. Start the Malmo server:
   ```bash
   cd graph-rl
   python graphrl/environments/malmo/malmo_env_env.py RUN_SERVER <port> 'minecraft/MalmoPlatform'
   ```

## Configuration Files

- **env_configs/**: Environment configurations (grid size, objectives, etc.)
- **model_configs/**: Neural network architectures and configurations
- **hyper_configs/**: Training hyperparameters (learning rate, batch size, etc.)
- **Knowledge graph files**: JSON files defining semantic relationships (e.g., pickaxe_skyline.json)

## Key Features

1. **Graph-Based State Embeddings**: Uses GCNs to embed game states with semantic knowledge
2. **Knowledge Graph Integration**: Incorporates structured knowledge about object relationships
3. **Linguistic Priors**: Leverages WordNet and other linguistic resources
4. **Modular Architecture**: Easily swap between baseline and graph-enhanced models
5. **Minecraft Integration**: Complex 3D environments for testing generalization

## Research Goals

- Investigate how linguistic and semantic knowledge improves RL agent performance
- Study generalization capabilities when agents have access to structured knowledge
- Explore efficient ways to integrate graph neural networks with RL algorithms
- Develop methods for automatic knowledge graph construction from linguistic resources

## Contributing

When contributing to this project:
1. Follow the existing code structure and naming conventions
2. Document new features and experiments
3. Add appropriate configuration files for new experiments
4. Update this README with any new setup requirements

## License

This project is licensed under a Restrictive Academic License - see the [LICENSE](LICENSE) file for details.

**Key Points:**
- Academic use only (no commercial use)
- No redistribution without permission
- Attribution required in publications
- For commercial licensing, contact [your-email@institution.edu]

## Citation

If you use this code in your research, please cite:
```bibtex
[Add citation information]
```

## Contact

[Add contact information]

## Acknowledgments

- Intel AI for the Graph-RL framework
- Microsoft for the Malmo platform


