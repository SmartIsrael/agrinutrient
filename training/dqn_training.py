import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym

from environment.custom_env import HydroponicEnv

def train_dqn_agent(total_timesteps=100000, eval_freq=10000, n_eval_episodes=5, save_path="models/dqn"):
    """
    Train a DQN agent on the HydroponicEnv environment.
    
    Args:
        total_timesteps (int): Total number of timesteps to train for
        eval_freq (int): Evaluation frequency in timesteps
        n_eval_episodes (int): Number of episodes to evaluate on
        save_path (str): Path to save the models
    
    Returns:
        model: Trained DQN model
        rewards: List of rewards during training
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
        name_prefix="dqn_hydroponics"
    )
    
    # Initialize the DQN agent
    model = DQN(
        "MlpPolicy", 
        env, 
        learning_rate=0.0001,
        buffer_size=100000,  # Experience replay buffer size
        learning_starts=1000,  # How many steps before starting learning
        batch_size=64,
        gamma=0.99,  # Discount factor
        exploration_fraction=0.2,  # Exploration vs exploitation trade-off
        exploration_initial_eps=1.0,  # Initial exploration rate
        exploration_final_eps=0.05,  # Final exploration rate
        train_freq=4,  # Update the model every 4 steps
        gradient_steps=1,  # How many gradient steps after each update
        target_update_interval=1000,  # Update the target network every 1000 steps
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

def plot_training_results(model, save_path="models/dqn"):
    """
    Plot training results for a trained model.
    
    Args:
        model: Trained model
        save_path (str): Path to save the plots
    """
    # Extract episode rewards from monitor
    rewards = model.ep_reward_buffer
    
    # Plot rewards over time
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards')
    plt.grid(True)
    plt.savefig(f"{save_path}/training_rewards.png")
    plt.close()

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
    model = train_dqn_agent(total_timesteps=100000)
    
    # Plot training results
    plot_training_results(model)
    
    # Demo the trained agent
    demo_trained_agent(model)