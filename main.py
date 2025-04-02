import argparse
import os
import gymnasium as gym
from stable_baselines3 import DQN, PPO

from environment.custom_env import HydroponicEnv
from training.dqn_training import train_dqn_agent, demo_trained_agent as demo_dqn
from training.pg_training import train_pg_agent, demo_trained_agent as demo_ppo, analyze_policy

def main():
    """
    Main entry point for the Hydroponic Nutrient Optimization RL project.
    """
    parser = argparse.ArgumentParser(description='Hydroponic Nutrient Optimization with RL')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'demo', 'analyze'],
                        help='Mode: train, test, demo, or analyze')
    
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'ppo'],
                        help='RL algorithm to use: dqn or ppo')
    
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps for training')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for test/demo/analyze modes)')
    
    args = parser.parse_args()
    
    # Ensure model directories exist
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)
    
    if args.mode == 'train':
        print(f"Training a {args.algorithm.upper()} agent for {args.timesteps} timesteps...")
        
        if args.algorithm == 'dqn':
            model = train_dqn_agent(total_timesteps=args.timesteps)
        else:  # ppo
            model = train_pg_agent(total_timesteps=args.timesteps)
            
        print("Training completed!")
    
    elif args.mode in ['test', 'demo', 'analyze']:
        if args.model_path is None:
            # Use default paths if not specified
            if args.algorithm == 'dqn':
                args.model_path = "models/dqn/final_model.zip"
            else:  # ppo
                args.model_path = "models/pg/final_model.zip"
        
        print(f"Loading model from {args.model_path}...")
        
        try:
            if args.algorithm == 'dqn':
                model = DQN.load(args.model_path)
            else:  # ppo
                model = PPO.load(args.model_path)
                
            print("Model loaded successfully!")
            
            if args.mode == 'test':
                # Test the model
                env = HydroponicEnv()
                obs, info = env.reset()
                
                # Run for one episode
                done = False
                total_reward = 0
                
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = env.step(action)
                    total_reward += reward
                    
                    if done or truncated:
                        break
                
                print(f"Test episode completed with total reward: {total_reward:.2f}")
                
            elif args.mode == 'demo':
                # Demo with visualization
                if args.algorithm == 'dqn':
                    demo_dqn(model)
                else:  # ppo
                    demo_ppo(model)
                    
            elif args.mode == 'analyze':
                # Analyze policy (only for PPO)
                if args.algorithm == 'ppo':
                    analyze_policy(model)
                else:
                    print("Policy analysis is only available for PPO algorithm.")
                    
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading model: {e}")
            print("Make sure you've trained a model first or# filepath: /home/begati/Desktop/agrinutrient/main.py specify a valid path.")
    else:
        print("Invalid mode. Choose from 'train', 'test', 'demo', or 'analyze'.")
    # Clean up
    env.close()
    print("Environment closed.")
    print("Exiting the program.")
    # Close the environment
    gym.envs.registration.registry.clear()
    gym.envs.registration.load_all()
    gym.utils.seeding.np_random.seed(0)
    gym.utils.seeding.np_random.get_state()

    
