import argparse
import os
import gymnasium as gym
import numpy as np
import json
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from environment.custom_env import HydroponicEnv
from training.dqn_training import (
    train_dqn_agent, 
    demo_trained_agent as demo_dqn,
    run_hyperparameter_optimization as run_dqn_optimization,
    train_with_best_params as train_dqn_with_best_params
)
from training.pg_training import (
    train_pg_agent, 
    demo_trained_agent as demo_ppo, 
    analyze_policy,
    run_hyperparameter_optimization as run_ppo_optimization,
    train_with_best_params as train_ppo_with_best_params
)
from utils.helpers import record_video, record_advanced_video

def main():
    """
    Main entry point for the Hydroponic Nutrient Optimization RL project.
    """
    print("Starting Hydroponic RL System...")
    
    parser = argparse.ArgumentParser(description='Hydroponic Nutrient Optimization with RL')
    
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test', 'demo', 'analyze', 'record', 'optimize', 'train_optimal'],
                        help='Mode: train, test, demo, analyze, record, optimize, or train_optimal')
    
    parser.add_argument('--algorithm', type=str, default='dqn',
                        choices=['dqn', 'ppo'],
                        help='RL algorithm to use: dqn or ppo')
    
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps for training')
    
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to saved model (for test/demo/analyze modes)')
    
    parser.add_argument('--video_path', type=str, default='videos/simulation.mp4',
                        help='Path to save the recorded video (only for record mode)')
    
    parser.add_argument('--advanced', action='store_true',
                        help='Use advanced video recording with metrics visualization')
    
    parser.add_argument('--n_trials', type=int, default=20,
                        help='Number of optimization trials to run (for optimize mode)')
    
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name of the Optuna study (defaults to "dqn_optimization" or "ppo_optimization")')
    
    args = parser.parse_args()
    
    # Set default study name based on algorithm if not provided
    if args.study_name is None:
        args.study_name = f"{args.algorithm}_optimization"
    
    # Ensure model directories exist
    os.makedirs("models/dqn", exist_ok=True)
    os.makedirs("models/pg", exist_ok=True)
    os.makedirs("models/dqn_tuned", exist_ok=True)
    os.makedirs("models/pg_tuned", exist_ok=True)
    
    # Track if we create an environment that needs to be closed
    env_created = False
    
    if args.mode == 'train':
        print(f"Training a {args.algorithm.upper()} agent for {args.timesteps} timesteps...")
        
        if args.algorithm == 'dqn':
            model = train_dqn_agent(total_timesteps=args.timesteps)
        else:  # ppo
            model = train_pg_agent(total_timesteps=args.timesteps)
            
        print("Training completed!")
    
    elif args.mode == 'optimize':
        print(f"Running hyperparameter optimization for {args.algorithm.upper()} with {args.n_trials} trials...")
        print("Note: Make sure you have Optuna installed ('pip install optuna')")
        
        try:
            import optuna
            if args.algorithm == 'dqn':
                best_params = run_dqn_optimization(
                    n_trials=args.n_trials, 
                    study_name=args.study_name
                )
            else:  # ppo
                best_params = run_ppo_optimization(
                    n_trials=args.n_trials, 
                    study_name=args.study_name
                )
                
            print("\nOptimization complete!")
            print("You can now use '--mode train_optimal' to train a model with these parameters.")
        except ImportError:
            print("Optuna is not installed. Please install it with 'pip install optuna'")
    
    elif args.mode == 'train_optimal':
        print(f"Training {args.algorithm.upper()} model with optimal hyperparameters...")
        
        try:
            if args.algorithm == 'dqn':
                model = train_dqn_with_best_params(total_timesteps=args.timesteps)
            else:  # ppo
                model = train_ppo_with_best_params(total_timesteps=args.timesteps)
                
            print("Training with optimal hyperparameters completed!")
            
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"Run hyperparameter optimization first with '--mode optimize --algorithm {args.algorithm}'")
    
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
                env_created = True
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
            print("Make sure you've trained a model first or specify a valid path.")
    
    elif args.mode == 'record':
        # Check if model path is specified
        if args.model_path is None:
            # Use default paths if not specified
            if args.algorithm == 'dqn':
                args.model_path = "models/dqn/final_model.zip"
            else:  # ppo
                args.model_path = "models/pg/final_model.zip"
                
        print(f"Loading model from {args.model_path} for video recording...")
        
        try:
            # Load the model
            if args.algorithm == 'dqn':
                model = DQN.load(args.model_path)
            else:  # ppo
                model = PPO.load(args.model_path)
                
            print("Recording simulation video...")
            
            # Ensure videos directory exists
            os.makedirs(os.path.dirname(args.video_path), exist_ok=True)
            
            if args.advanced:
                record_advanced_video(model, HydroponicEnv, args.video_path)
            else:
                record_video(model, HydroponicEnv, args.video_path)
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Error loading model: {e}")
            print("Make sure you've trained a model first or specify a valid path.")
    
    else:
        print("Invalid mode. Choose from 'train', 'test', 'demo', 'analyze', 'record', 'optimize', or 'train_optimal'.")
    
    # Clean up only if we created an environment
    if env_created:
        env.close()
        print("Environment closed.")
    
    print("Exiting the program.")

if __name__ == "__main__":
    main()


