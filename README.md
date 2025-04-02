# Hydroponic Nutrient Optimization using Reinforcement Learning

This project implements a Reinforcement Learning (RL) system to optimize nutrient delivery in hydroponic farming systems. The RL agent learns to dynamically adjust pH levels, Electrical Conductivity (EC), and water cycles to maintain optimal growing conditions for plants at different growth stages.

## Project Structure

```
project_root/
├── environment/
│   ├── custom_env.py          # Custom Gym environment (Hydroponics simulation)
│   ├── rendering.py           # Visualization using PyGame
├── training/
│   ├── dqn_training.py        # DQN training script using Stable-Baselines3
│   ├── pg_training.py         # PPO training script using Stable-Baselines3
├── models/
│   ├── dqn/                   # Saved DQN models
│   ├── pg/                    # Saved policy gradient models
├── utils/
│   ├── helpers.py             # Helper functions for analysis and visualization
├── main.py                    # Entry point for running RL experiments
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```

## Features

- Custom Gymnasium environment simulating hydroponic systems
- State space includes pH, EC, temperature, and plant growth stage
- Action space allows for adjusting nutrients, pH, and water cycle timing
- Two RL algorithms implemented: DQN and PPO (Proximal Policy Optimization)
- Visual rendering of the hydroponic system and plant growth
- Detailed analysis tools for policy exploration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hydroponic-rl.git
cd hydroponic-rl

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training a new agent

```bash
# Train a DQN agent
python main.py --mode train --algorithm dqn --timesteps 100000

# Train a PPO agent
python main.py --mode train --algorithm ppo --timesteps 100000
```

### Testing a trained agent

```bash
# Test a trained DQN agent
python main.py --mode test --algorithm dqn --model_path models/dqn/final_model.zip

# Test a trained PPO agent
python main.py --mode test --algorithm ppo --model_path models/pg/final_model.zip
```

### Demonstrating a trained agent with visualization

```bash
# Demo a trained DQN agent
python main.py --mode demo --algorithm dqn --model_path models/dqn/final_model.zip

# Demo a trained PPO agent
python main.py --mode demo --algorithm ppo --model_path models/pg/final_model.zip
```

### Analyzing a trained policy (PPO only)

```bash
# Analyze a trained PPO policy
python main.py --mode analyze --algorithm ppo --model_path models/pg/final_model.zip
```

## Learning Process

The RL agent learns through trial and error by interacting with the environment:

1. The agent observes the current state of the hydroponic system
2. It selects an action (adjust nutrients, pH, or water cycle)
3. The environment simulates the effect of this action
4. The agent receives a reward based on how optimal the conditions are
5. Over time, the agent learns to maximize rewards by maintaining ideal growing conditions

## License

MIT# agrinutrient
# agrinutrient
# agrinutrient
# agrinutrient
# agrinutrient
