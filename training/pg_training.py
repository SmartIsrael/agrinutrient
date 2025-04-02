import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym

from environment.custom_env import HydroponicEnv

def train_pg_agent(total_timesteps=100000, eval_freq=10000, n_eval_episodes=5, save_path="models/pg"):
    """
    Train a PPO (Proximal Policy Optimization) agent on the HydroponicEnv environment.
    
    Args:
        total_timesteps (int): Total number of timesteps to train for
        eval_freq (int): Evaluation frequency in timesteps
        n_eval_episodes (int): Number of episodes to evaluate on
        save_path (str): Path to save the models
    
    Returns:
        model: Trained PPO model
    """
    # Create directories if they don't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Create and wrap the environment
    env = HydroponicEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = HydroponicEnv()
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best/",
        log_path=f"{save_path}/logs/",
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=f"{save_path}/checkpoints/",
        name_prefix="ppo_hydroponics"
    )
    
    # Initialize the PPO agent
    model = PPO(
        "MlpPolicy", 
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Number of steps to run for each environment per update
        batch_size=64,  # Minibatch size
        n_epochs=10,  # Number of epochs when optimizing the surrogate loss
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        clip_range=0.2,  # Clipping parameter for PPO
        clip_range_vf=None,  # Clipping parameter for the value function (None = no clipping)
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,  # Value function coefficient
        max_grad_norm=0.5,  # Maximum value for gradient clipping
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard/"
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )
    
    # Save the final model
    model.save(f"{save_path}/final_model")
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model

def analyze_policy(model, save_path="models/pg"):
    """
    Analyze the policy learned by the PPO agent.
    
    Args:
        model: Trained PPO model
        save_path (str): Path to save the analysis
    """
    env = HydroponicEnv()
    
    # Sample different observations and analyze actions
    states = []
    actions = []
    
    # Generate a range of pH and EC values to analyze
    ph_values = np.linspace(4.0, 8.0, 9)
    ec_values = np.linspace(0.5, 3.5, 7)
    
    # For a fixed temperature and growth stage
    temperature = 22.0
    growth_stage = 1  # Vegetative stage
    
    # Create a matrix to store action probabilities for each state
    action_map = np.zeros((len(ph_values), len(ec_values)), dtype=int)
    
    # Analyze policy for different pH and EC combinations
    for i, ph in enumerate(ph_values):
        for j, ec in enumerate(ec_values):
            obs = np.array([ph, ec, temperature, growth_stage], dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            action_map[i, j] = action
            states.append(obs)
            actions.append(action)
    
    # Plot policy map for pH vs EC
    plt.figure(figsize=(12, 10))
    plt.imshow(action_map, origin='lower', interpolation='nearest', aspect='auto')
    plt.colorbar(label='Action')
    plt.xticks(np.arange(len(ec_values)), [f"{ec:.1f}" for ec in ec_values])
    plt.yticks(np.arange(len(ph_values)), [f"{ph:.1f}" for ph in ph_values])
    plt.xlabel('EC (mS/cm)')
    plt.ylabel('pH')
    plt.title(f'PPO Policy Map (Temperature={temperature}, Growth Stage={growth_stage})')
    plt.savefig(f"{save_path}/policy_map.png")
    plt.close()
    
    # Print action interpretations
    action_meanings = [
        "Decrease nutrient", "No change (nutrients)", "Increase nutrient",
        "Decrease pH", "No change (pH)", "Increase pH",
        "Decrease water cycle", "No change (water cycle)", "Increase water cycle"
    ]
    
    print("Action interpretations:")
    for i, meaning in enumerate(action_meanings):
        print(f"Action {i}: {meaning}")

def demo_trained_agent(model, num_episodes=3):
    """
    Run a demonstration of the trained agent.
    
    Args:
        model: Trained model
        num_episodes (int): Number of episodes to run
    """
    env = HydroponicEnv(render_mode="human")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            env.render()
            
            # Print state information
            print(f"Step {steps}, Action: {action}, Reward: {reward:.2f}")
            print(f"  pH: {info['ph']:.2f}, EC: {info['ec']:.2f}, Temp: {info['temperature']:.2f}")
            print(f"  Growth Stage: {info['growth_stage']}, Day: {info['total_days']}")
            
            if done or truncated:
                break
        
        print(f"Episode {episode+1} finished with total reward: {total_reward:.2f} after {steps} steps")
    
    env.close()

if __name__ == "__main__":
    # Train the agent
    model = train_pg_agent(total_timesteps=100000)
    
    # Analyze the learned policy
    analyze_policy(model)
    
    # Demo the trained agent
    demo_trained_agent(model)