import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import load_monitor_data

def plot_learning_curve(log_folder, title='Learning Curve'):
    """
    Plot the training reward curve from monitor logs.
    
    Args:
        log_folder (str): Path to the monitor logs
        title (str): Title of the plot
    """
    data = load_monitor_data(os.path.join(log_folder, "*.monitor.csv"))
    
    if isinstance(data, pd.DataFrame):
        y = data["r"]
        x = np.arange(len(y))
    else:
        # Handle legacy monitor data
        x, y = ts2xy(load_results(log_folder), 'timesteps')
    
    # Ensure data is not empty
    if len(x) > 0:
        # Smoothed curve
        window_size = min(50, len(x)//10) if len(x) > 100 else 1
        if window_size > 1:
            y_smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
            x_smoothed = x[window_size-1:]
        else:
            y_smoothed, x_smoothed = y, x
        
        # Create plot
        plt.figure(figsize=(12, 6))
        plt.scatter(x, y, s=2, alpha=0.3, label='Rewards')
        if window_size > 1:
            plt.plot(x_smoothed, y_smoothed, linewidth=2, label=f'Smoothed (window={window_size})')
        plt.xlabel('Timesteps')
        plt.ylabel('Rewards')
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        results_path = os.path.join(log_folder, "learning_curve.png")
        plt.savefig(results_path)
        plt.close()
        print(f"Learning curve saved to {results_path}")
    else:
        print("No data available for plotting learning curve")

def analyze_environment_metrics(log_folder, num_episodes=10):
    """
    Analyzes metrics recorded during environment evaluation.
    
    Args:
        log_folder (str): Path to the evaluation metrics
        num_episodes (int): Number of recent episodes to analyze
    """
    try:
        # Load metrics from evaluation runs
        metrics_file = os.path.join(log_folder, "metrics.npz")
        if os.path.exists(metrics_file):
            data = np.load(metrics_file, allow_pickle=True)
            episode_metrics = data['episode_metrics'].item()
            
            # Extract relevant metrics
            ph_values = episode_metrics.get('ph', [])
            ec_values = episode_metrics.get('ec', [])
            temp_values = episode_metrics.get('temperature', [])
            rewards = episode_metrics.get('rewards', [])
            
            # Plot metrics
            if len(ph_values) > 0 and len(ec_values) > 0:
                plt.figure(figsize=(12, 8))
                
                # Take only the most recent episodes if there are many
                recent_n = min(num_episodes, len(ph_values))
                x = np.arange(recent_n)
                
                plt.subplot(2, 2, 1)
                plt.plot(x, ph_values[-recent_n:])
                plt.axhline(y=5.5, color='g', linestyle='--')
                plt.axhline(y=7.0, color='g', linestyle='--')
                plt.title('pH Levels')
                plt.xlabel('Steps')
                plt.ylabel('pH')
                
                plt.subplot(2, 2, 2)
                plt.plot(x, ec_values[-recent_n:])
                plt.axhline(y=1.0, color='g', linestyle='--')
                plt.axhline(y=3.0, color='g', linestyle='--')
                plt.title('EC Levels')
                plt.xlabel('Steps')
                plt.ylabel('EC (mS/cm)')
                
                plt.subplot(2, 2, 3)
                plt.plot(x, temp_values[-recent_n:])
                plt.axhline(y=18.0, color='g', linestyle='--')
                plt.axhline(y=26.0, color='g', linestyle='--')
                plt.title('Temperature')
                plt.xlabel('Steps')
                plt.ylabel('Temperature (°C)')
                
                plt.subplot(2, 2, 4)
                plt.plot(x, rewards[-recent_n:])
                plt.title('Rewards')
                plt.xlabel('Steps')
                plt.ylabel('Reward')
                
                plt.tight_layout()
                plot_path = os.path.join(log_folder, "metrics_analysis.png")
                plt.savefig(plot_path)
                plt.close()
                print(f"Metrics analysis saved to {plot_path}")
            else:
                print("Insufficient data for metrics analysis")
        else:
            print(f"No metrics file found at {metrics_file}")
    except Exception as e:
        print(f"Error analyzing environment metrics: {e}")

def create_requirements_file(output_path="requirements.txt"):
    """
    Creates a requirements.txt file with the necessary dependencies.
    
    Args:
        output_path (str): Path to save the requirements.txt file
    """
    requirements = [
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "pandas>=1.3.0",
        "gymnasium>=0.26.0",
        "stable-baselines3>=1.6.0",
        "pygame>=2.1.0",
        "torch>=1.11.0",
    ]
    
    with open(output_path, "w") as f:
        for req in requirements:
            f.write(f"{req}\n")
    
    print(f"Requirements file created at {output_path}")

if __name__ == "__main__":
    # Example usage
    # create_requirements_file()
    # plot_learning_curve("models/dqn/logs")
    pass