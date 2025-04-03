import os
import json
import optuna
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import gymnasium as gym

from environment.custom_env import HydroponicEnv

def train_dqn_agent(total_timesteps=100000, eval_freq=10000, n_eval_episodes=5, save_path="models/dqn", custom_model=None):
    """
    Train a DQN agent on the HydroponicEnv environment.
    
    Args:
        total_timesteps (int): Total number of timesteps to train for
        eval_freq (int): Evaluation frequency in timesteps
        n_eval_episodes (int): Number of episodes to evaluate on
        save_path (str): Path to save the models
        custom_model (DQN, optional): Pre-configured DQN model with custom hyperparameters
    
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
    
    # Initialize the DQN agent if not provided
    if custom_model is None:
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
            tensorboard_log=f"{save_path}/tensorboard/",
        )
    else:
        model = custom_model
    
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

def optimize_hyperparams_objective(trial):
    """Optuna objective function to minimize."""
    
    # Define the hyperparameters to tune
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
        'buffer_size': trial.suggest_int('buffer_size', 50000, 500000),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.9999),
        'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
        'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.1),
        'target_update_interval': trial.suggest_int('target_update_interval', 1000, 10000),
        'learning_starts': trial.suggest_int('learning_starts', 1000, 10000),
        'gradient_steps': trial.suggest_int('gradient_steps', 1, 10),
    }
    
    # Set up directories
    trial_dir = f"tuning_results/dqn/trial_{trial.number}"
    model_dir = f"{trial_dir}/model"
    tensorboard_dir = f"{trial_dir}/tensorboard"
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Save hyperparameters
    with open(f"{trial_dir}/hyperparams.json", 'w') as f:
        json.dump(hyperparams, f, indent=4)
    
    # Create environment
    env = HydroponicEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = HydroponicEnv()
    
    # Create the model with trial hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=hyperparams['learning_rate'],
        buffer_size=hyperparams['buffer_size'],
        batch_size=hyperparams['batch_size'],
        gamma=hyperparams['gamma'],
        exploration_fraction=hyperparams['exploration_fraction'],
        exploration_final_eps=hyperparams['exploration_final_eps'],
        target_update_interval=hyperparams['target_update_interval'],
        learning_starts=hyperparams['learning_starts'],
        gradient_steps=hyperparams['gradient_steps'],
        verbose=0,
        tensorboard_log=tensorboard_dir
    )
    
    # Setup eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    try:
        # Train the model
        model.learn(
            total_timesteps=50000,  # Reduced from 100k for faster trials
            callback=eval_callback
        )
        
        # Evaluate the model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        
        # Optuna maximizes the objective function, so return the mean reward
        return mean_reward
    
    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf')  # Return negative infinity if trial fails
    
    finally:
        # Cleanup
        env.close()
        eval_env.close()

def run_hyperparameter_optimization(n_trials=20, study_name="dqn_optimization"):
    """Run the hyperparameter optimization."""
    
    # Create results directory
    os.makedirs("tuning_results/dqn", exist_ok=True)
    
    # Create a new study or load an existing one
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///tuning_results/dqn/{study_name}.db",
        load_if_exists=True
    )
    
    # Get the number of trials already completed
    n_completed = len(study.trials)
    print(f"Study already has {n_completed} completed trials")
    
    # Calculate how many more trials to run
    n_remaining = max(0, n_trials - n_completed)
    print(f"Running {n_remaining} more trials to reach the target of {n_trials}")
    
    # Only run optimization if more trials are needed
    if n_remaining > 0:
        study.optimize(optimize_hyperparams_objective, n_trials=n_remaining)
    else:
        print(f"Already completed {n_completed} trials, no more trials needed")
    
    # Print results
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    # Save best hyperparameters
    best_params = trial.params
    with open("tuning_results/dqn/best_hyperparams.json", 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params

def train_with_best_params(total_timesteps=100000):
    """Train a model with the best hyperparameters found."""
    
    print(f"Training DQN model with the best hyperparameters for {total_timesteps} timesteps...")
    
    # Load best hyperparameters
    try:
        with open("tuning_results/dqn/best_hyperparams.json", 'r') as f:
            best_params = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("No best hyperparameters file found. Run hyperparameter optimization first.")
    
    # Create environment
    env = HydroponicEnv()
    env = Monitor(env)
    
    # Create evaluation environment
    eval_env = HydroponicEnv()
    
    # Create the model with best hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=best_params['learning_rate'],
        buffer_size=best_params['buffer_size'],
        batch_size=best_params['batch_size'],
        gamma=best_params['gamma'],
        exploration_fraction=best_params['exploration_fraction'],
        exploration_final_eps=best_params['exploration_final_eps'],
        target_update_interval=best_params['target_update_interval'],
        learning_starts=best_params['learning_starts'],
        gradient_steps=best_params['gradient_steps'],
        verbose=1,
        tensorboard_log="models/dqn_tuned/tensorboard/"
    )
    
    # Setup eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="models/dqn_tuned/best/",
        log_path="models/dqn_tuned/logs/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback
    )
    
    # Save the final model
    model.save("models/dqn/final_model")
    
    # Evaluate the final model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Final model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    # Cleanup
    env.close()
    eval_env.close()
    
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