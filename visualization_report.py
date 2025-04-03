# visualization_report.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import HydroponicEnv
from utils.helpers import evaluate_model

def plot_cumulative_rewards(num_episodes=5000):
    """
    Plot cumulative rewards over episodes for DQN and PPO models
    """
    # Ensure directories exist
    os.makedirs('report_plots', exist_ok=True)
    
    # Load models
    dqn_model = DQN.load("models/dqn/final_model.zip")
    ppo_model = PPO.load("models/pg/final_model.zip")
    
    # Run episodes for each model and collect rewards
    dqn_rewards = []
    ppo_rewards = []
    
    env = HydroponicEnv()
    
    # DQN evaluation
    print("Evaluating DQN model...")
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = dqn_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if done or truncated:
                break
        
        dqn_rewards.append(episode_reward)
        print(f"DQN Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    # PPO evaluation
    print("\nEvaluating PPO model...")
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            if done or truncated:
                break
        
        ppo_rewards.append(episode_reward)
        print(f"PPO Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    # Calculate cumulative rewards
    dqn_cumulative = np.cumsum(dqn_rewards)
    ppo_cumulative = np.cumsum(ppo_rewards)
    
    # Plot cumulative rewards
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, num_episodes + 1), dqn_cumulative, label='DQN', marker='o', linestyle='-', alpha=0.7)
    plt.plot(range(1, num_episodes + 1), ppo_cumulative, label='PPO', marker='x', linestyle='-', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Episodes')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('report_plots/cumulative_rewards.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot episode rewards
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, num_episodes + 1), dqn_rewards, label='DQN', marker='o', linestyle='-', alpha=0.7)
    plt.plot(range(1, num_episodes + 1), ppo_rewards, label='PPO', marker='x', linestyle='-', alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Rewards Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('report_plots/episode_rewards.png', dpi=300, bbox_inches='tight')
    
    print("Reward plots saved to report_plots/ directory")
    env.close()

def plot_training_stability():
    """
    Extract and plot loss curves for DQN and policy entropy for PPO
    from TensorBoard logs
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import glob
    except ImportError:
        print("Please install tensorboard: pip install tensorboard")
        return
    
    # Find TensorBoard log files
    dqn_log_path = glob.glob('models/dqn/tensorboard/*')
    ppo_log_path = glob.glob('models/pg/tensorboard/*')
    
    if not dqn_log_path or not ppo_log_path:
        print("TensorBoard logs not found. Please train the models first.")
        return
    
    # Extract DQN loss data
    dqn_event_acc = EventAccumulator(dqn_log_path[0])
    dqn_event_acc.Reload()
    
    # Extract PPO entropy data
    ppo_event_acc = EventAccumulator(ppo_log_path[0])
    ppo_event_acc.Reload()
    
    # Get available tags
    dqn_tags = dqn_event_acc.Tags()['scalars']
    ppo_tags = ppo_event_acc.Tags()['scalars']
    
    print("Available DQN metrics:", dqn_tags)
    print("Available PPO metrics:", ppo_tags)
    
    # Extract DQN loss values
    if 'train/loss' in dqn_tags:
        dqn_loss = dqn_event_acc.Scalars('train/loss')
        dqn_steps = [x.step for x in dqn_loss]
        dqn_loss_values = [x.value for x in dqn_loss]
        
        # Plot DQN loss
        plt.figure(figsize=(12, 6))
        plt.plot(dqn_steps, dqn_loss_values)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('DQN Training Loss')
        plt.grid(True, alpha=0.3)
        plt.savefig('report_plots/dqn_loss.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Extract PPO entropy values
    if 'train/entropy_loss' in ppo_tags:
        ppo_entropy = ppo_event_acc.Scalars('train/entropy_loss')
        ppo_steps = [x.step for x in ppo_entropy]
        ppo_entropy_values = [x.value for x in ppo_entropy]
        
        # Plot PPO entropy
        plt.figure(figsize=(12, 6))
        plt.plot(ppo_steps, ppo_entropy_values)
        plt.xlabel('Training Steps')
        plt.ylabel('Policy Entropy')
        plt.title('PPO Policy Entropy')
        plt.grid(True, alpha=0.3)
        plt.savefig('report_plots/ppo_entropy.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Also extract reward during training if available
    dqn_rewards = None
    ppo_rewards = None
    
    if 'rollout/ep_rew_mean' in dqn_tags:
        dqn_rewards_data = dqn_event_acc.Scalars('rollout/ep_rew_mean')
        dqn_reward_steps = [x.step for x in dqn_rewards_data]
        dqn_rewards = [x.value for x in dqn_rewards_data]
    
    if 'rollout/ep_rew_mean' in ppo_tags:
        ppo_rewards_data = ppo_event_acc.Scalars('rollout/ep_rew_mean')
        ppo_reward_steps = [x.step for x in ppo_rewards_data]
        ppo_rewards = [x.value for x in ppo_rewards_data]
    
    # Plot training rewards if available
    if dqn_rewards and ppo_rewards:
        plt.figure(figsize=(12, 6))
        plt.plot(dqn_reward_steps, dqn_rewards, label='DQN')
        plt.plot(ppo_reward_steps, ppo_rewards, label='PPO')
        plt.xlabel('Training Steps')
        plt.ylabel('Average Episode Reward')
        plt.title('Training Rewards')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('report_plots/training_rewards.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print("Training stability plots saved to report_plots/ directory")

def plot_episodes_to_convergence():
    """
    Analyze training logs to find the episodes where algorithms converged
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        import glob
    except ImportError:
        print("Please install tensorboard: pip install tensorboard")
        return
    
    # Find TensorBoard log files
    dqn_log_path = glob.glob('models/dqn/tensorboard/*')
    ppo_log_path = glob.glob('models/pg/tensorboard/*')
    
    if not dqn_log_path or not ppo_log_path:
        print("TensorBoard logs not found. Please train the models first.")
        return
    
    # Extract data
    dqn_event_acc = EventAccumulator(dqn_log_path[0])
    ppo_event_acc = EventAccumulator(ppo_log_path[0])
    
    dqn_event_acc.Reload()
    ppo_event_acc.Reload()
    
    # Extract reward data
    dqn_tags = dqn_event_acc.Tags()['scalars']
    ppo_tags = ppo_event_acc.Tags()['scalars']
    
    # Define convergence threshold (you might need to adjust this)
    convergence_threshold = 200  # Reward threshold for convergence
    
    # Extract rewards
    if 'rollout/ep_rew_mean' in dqn_tags and 'rollout/ep_rew_mean' in ppo_tags:
        dqn_rewards_data = dqn_event_acc.Scalars('rollout/ep_rew_mean')
        ppo_rewards_data = ppo_event_acc.Scalars('rollout/ep_rew_mean')
        
        dqn_steps = [x.step for x in dqn_rewards_data]
        dqn_rewards = [x.value for x in dqn_rewards_data]
        
        ppo_steps = [x.step for x in ppo_rewards_data]
        ppo_rewards = [x.value for x in ppo_rewards_data]
        
        # Find convergence points
        dqn_convergence_idx = None
        ppo_convergence_idx = None
        
        # Use a window of 10 episodes to determine stable convergence
        window_size = 10
        
        for i in range(len(dqn_rewards) - window_size):
            window_avg = np.mean(dqn_rewards[i:i+window_size])
            if window_avg >= convergence_threshold:
                dqn_convergence_idx = i
                break
        
        for i in range(len(ppo_rewards) - window_size):
            window_avg = np.mean(ppo_rewards[i:i+window_size])
            if window_avg >= convergence_threshold:
                ppo_convergence_idx = i
                break
        
        # Plot convergence
        plt.figure(figsize=(12, 8))
        plt.plot(dqn_steps, dqn_rewards, label='DQN', alpha=0.7)
        plt.plot(ppo_steps, ppo_rewards, label='PPO', alpha=0.7)
        
        # Mark convergence points
        if dqn_convergence_idx is not None:
            plt.axvline(x=dqn_steps[dqn_convergence_idx], color='blue', linestyle='--', 
                        label=f'DQN Convergence: {dqn_steps[dqn_convergence_idx]} steps')
        
        if ppo_convergence_idx is not None:
            plt.axvline(x=ppo_steps[ppo_convergence_idx], color='orange', linestyle='--',
                        label=f'PPO Convergence: {ppo_steps[ppo_convergence_idx]} steps')
        
        plt.xlabel('Training Steps')
        plt.ylabel('Average Episode Reward')
        plt.title('Convergence Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('report_plots/convergence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Convergence analysis saved to report_plots/convergence_analysis.png")
        
        # Print convergence information
        if dqn_convergence_idx is not None:
            print(f"DQN converged after {dqn_steps[dqn_convergence_idx]} steps")
        else:
            print("DQN did not reach convergence threshold")
            
        if ppo_convergence_idx is not None:
            print(f"PPO converged after {ppo_steps[ppo_convergence_idx]} steps")
        else:
            print("PPO did not reach convergence threshold")
    else:
        print("Required metrics not found in TensorBoard logs")
        
def generate_report_visualizations():
    """Generate all visualizations for the report"""
    os.makedirs('report_plots', exist_ok=True)
    
    print("Generating cumulative reward plots...")
    plot_cumulative_rewards(num_episodes=5000)
    
    print("\nGenerating training stability plots...")
    plot_training_stability()
    
    print("\nGenerating convergence analysis...")
    plot_episodes_to_convergence()
    
    # Run comprehensive evaluation for parameter comparison
    print("\nGenerating comprehensive evaluation and parameter comparison...")
    from comprehensive_eval import compare_algorithms
    compare_algorithms()
    
    print("\nAll report visualizations generated!")

if __name__ == "__main__":
    generate_report_visualizations()