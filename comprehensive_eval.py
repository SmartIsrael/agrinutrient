# comprehensive_eval.py
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import DQN, PPO
from environment.custom_env import HydroponicEnv
from utils.helpers import analyze_environment_metrics

def comprehensive_evaluation(model_path, algorithm="dqn", num_episodes=20):
    """Run a comprehensive evaluation of the model"""
    # Load model
    if algorithm == "dqn":
        model = DQN.load(model_path)
    else:
        model = PPO.load(model_path)
    
    # Run evaluation episodes
    env = HydroponicEnv()
    
    # Track performance metrics
    rewards = []
    ph_in_range = []  # percentage of time pH was in optimal range
    ec_in_range = []  # percentage of time EC was in optimal range
    temp_in_range = [] # percentage of time temperature was in optimal range
    
    for i in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        # Track optimal ranges per step
        ph_optimal_steps = 0
        ec_optimal_steps = 0
        temp_optimal_steps = 0
        total_steps = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            total_steps += 1
            
            # Check if parameters are in optimal range based on growth stage
            growth_stage = int(info['growth_stage'])
            optimal = env.optimal_ranges[growth_stage]
            
            if optimal[0] <= info['ph'] <= optimal[1]:
                ph_optimal_steps += 1
            
            if optimal[2] <= info['ec'] <= optimal[3]:
                ec_optimal_steps += 1
                
            if optimal[4] <= info['temperature'] <= optimal[5]:
                temp_optimal_steps += 1
            
            if done or truncated:
                break
        
        rewards.append(episode_reward)
        
        # Calculate percentages
        if total_steps > 0:
            ph_in_range.append(ph_optimal_steps / total_steps * 100)
            ec_in_range.append(ec_optimal_steps / total_steps * 100)
            temp_in_range.append(temp_optimal_steps / total_steps * 100)
        
        print(f"Episode {i+1}/{num_episodes}: Reward = {episode_reward:.2f}")
    
    # Print overall statistics
    print("\nOverall Performance:")
    print(f"Mean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"pH in optimal range: {np.mean(ph_in_range):.1f}% ± {np.std(ph_in_range):.1f}%")
    print(f"EC in optimal range: {np.mean(ec_in_range):.1f}% ± {np.std(ec_in_range):.1f}%")
    print(f"Temperature in optimal range: {np.mean(temp_in_range):.1f}% ± {np.std(temp_in_range):.1f}%")
    
    # Determine the correct directory path
    # Map 'ppo' to 'pg' for directory structure compatibility
    dir_name = algorithm
    if algorithm == "ppo":
        dir_name = "pg"
    
    # Create directory if it doesn't exist
    os.makedirs(f'models/{dir_name}', exist_ok=True)
    
    # Create a bar chart of parameter optimization success
    plt.figure(figsize=(10, 6))
    plt.bar(['pH', 'EC', 'Temperature'], 
            [np.mean(ph_in_range), np.mean(ec_in_range), np.mean(temp_in_range)],
            yerr=[np.std(ph_in_range), np.std(ec_in_range), np.std(temp_in_range)])
    plt.ylim(0, 100)
    plt.ylabel('% of Time in Optimal Range')
    plt.title(f'{algorithm.upper()} Model Performance Metrics')
    plt.savefig(f'models/{dir_name}/performance_metrics.png')
    
    env.close()
    return rewards, ph_in_range, ec_in_range, temp_in_range

def compare_algorithms():
    """Compare DQN vs PPO performance"""
    print("Comparing DQN and PPO algorithms...")
    
    # Evaluate DQN
    print("\n=== Evaluating DQN Model ===")
    dqn_rewards, dqn_ph, dqn_ec, dqn_temp = comprehensive_evaluation(
        "models/dqn/final_model.zip", "dqn", num_episodes=10)
    
    # Evaluate PPO
    print("\n=== Evaluating PPO Model ===")
    ppo_rewards, ppo_ph, ppo_ec, ppo_temp = comprehensive_evaluation(
        "models/pg/final_model.zip", "ppo", num_episodes=10)
    
    # Ensure the directory exists
    os.makedirs('models', exist_ok=True)
    
    # Create comparison chart
    plt.figure(figsize=(12, 10))
    
    # Comparison of mean rewards
    plt.subplot(2, 1, 1)
    plt.bar(['DQN', 'PPO'], 
            [np.mean(dqn_rewards), np.mean(ppo_rewards)],
            yerr=[np.std(dqn_rewards), np.std(ppo_rewards)])
    plt.ylabel('Mean Episode Reward')
    plt.title('Reward Comparison Between DQN and PPO')
    
    # Comparison of parameters in optimal range
    plt.subplot(2, 1, 2)
    
    x = np.arange(3)
    width = 0.35
    
    dqn_means = [np.mean(dqn_ph), np.mean(dqn_ec), np.mean(dqn_temp)]
    ppo_means = [np.mean(ppo_ph), np.mean(ppo_ec), np.mean(ppo_temp)]
    
    dqn_std = [np.std(dqn_ph), np.std(dqn_ec), np.std(dqn_temp)]
    ppo_std = [np.std(ppo_ph), np.std(ppo_ec), np.std(ppo_temp)]
    
    plt.bar(x - width/2, dqn_means, width, label='DQN', yerr=dqn_std)
    plt.bar(x + width/2, ppo_means, width, label='PPO', yerr=ppo_std)
    plt.ylabel('% of Time in Optimal Range')
    plt.title('Parameter Optimization Comparison')
    plt.xticks(x, ['pH', 'EC', 'Temperature'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/algorithm_comparison.png')
    plt.close()
    
    print("\nComparison completed and saved to models/algorithm_comparison.png")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RL models for hydroponic nutrient optimization')
    parser.add_argument('--algorithm', type=str, default='both', choices=['dqn', 'ppo', 'both'],
                      help='Algorithm to evaluate: dqn, ppo, or both')
    parser.add_argument('--episodes', type=int, default=10,
                      help='Number of episodes to evaluate')
    
    args = parser.parse_args()
    
    if args.algorithm == 'both':
        compare_algorithms()
    else:
        # Determine the correct directory path
        dir_name = 'dqn' if args.algorithm == 'dqn' else 'pg'
        model_path = f"models/{dir_name}/final_model.zip"
        comprehensive_evaluation(model_path, args.algorithm, args.episodes)