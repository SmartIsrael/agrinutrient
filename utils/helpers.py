import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

# Replace the problematic import
# from stable_baselines3.common.monitor import load_monitor_data
# with our own implementation:

def load_monitor_data(path_glob):
    """Custom implementation of load_monitor_data since it's missing in your SB3 version."""
    monitor_files = glob.glob(path_glob)
    if not monitor_files:
        print(f"No monitor files found at {path_glob}")
        return pd.DataFrame()
    
    data_frames = []
    for file_path in monitor_files:
        try:
            data_frame = pd.read_csv(file_path, skiprows=1)
            data_frames.append(data_frame)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    
    if data_frames:
        data = pd.concat(data_frames)
        data.sort_values('t', inplace=True)
        data.reset_index(drop=True, inplace=True)
        return data
    return pd.DataFrame()

# Keep the other imports
from stable_baselines3.common.results_plotter import load_results, ts2xy
import imageio
import time
import cv2
import io

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

def record_video(model, env_class, video_path="videos/simulation.mp4", episode_length=600, width=800, height=600, fps=30):
    """
    Records a video of the agent interacting with the environment.
    
    Args:
        model: Trained RL model
        env_class: Environment class to instantiate
        video_path (str): Path to save the video
        episode_length (int): Maximum number of steps in the episode
        width (int): Width of the video
        height (int): Height of the video
        fps (int): Frames per second
    """
    # Create directory for videos if it doesn't exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Create environment with rgb_array rendering
    env = env_class(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Initialize video writer
    frames = []
    
    # Run simulation
    done = False
    step_count = 0
    total_reward = 0
    
    print(f"Recording video to {video_path}...")
    
    while not done and step_count < episode_length:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Get rendered frame
        frame = env.render()
        
        # Add step and reward info to the frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            frame,
            f"Step: {step_count} | Reward: {reward:.2f} | Total: {total_reward:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Add to frames
        frames.append(frame)
        
        # Increment step counter
        step_count += 1
        
        if done or truncated:
            break
    
    # Close environment
    env.close()
    
    # Save video
    if frames:
        print(f"Saving video with {len(frames)} frames at {fps} FPS...")
        imageio.mimsave(video_path, frames, fps=fps)
        print(f"Video saved to {video_path}")
    else:
        print("No frames captured, video not saved.")

def record_advanced_video(model, env_class, video_path="videos/advanced_simulation.mp4", episode_length=600, fps=30):
    """
    Records an advanced video of the agent interacting with the environment with additional visualizations.
    """
    # Create directory for videos if it doesn't exist
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    
    # Create environment with rgb_array rendering
    env = env_class(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Initialize video writer
    frames = []
    
    # Action meanings for display
    action_meanings = [
        "Decrease nutrient", "No change (nutrients)", "Increase nutrient",
        "Decrease pH", "No change (pH)", "Increase pH",
        "Decrease water cycle", "No change (water cycle)", "Increase water cycle"
    ]
    
    # For plotting history
    history = {
        'ph': [],
        'ec': [],
        'temp': [],
        'reward': [],
        'growth_stage': []
    }
    
    # Run simulation
    done = False
    step_count = 0
    total_reward = 0
    
    print(f"Recording advanced video to {video_path}...")
    
    while not done and step_count < episode_length:
        # Get action from model
        action, _states = model.predict(obs, deterministic=True)
        
        # Step the environment
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        
        # Update history
        history['ph'].append(info['ph'])
        history['ec'].append(info['ec'])
        history['temp'].append(info['temperature'])
        history['reward'].append(reward)
        history['growth_stage'].append(info['growth_stage'])
        
        # Get rendered frame
        frame = env.render()
        
        # Create an enhanced frame with metrics
        enhanced_frame = np.zeros((800, 1200, 3), dtype=np.uint8)
        enhanced_frame.fill(255)  # White background
        
        # Copy the original rendering
        enhanced_frame[0:600, 0:800] = frame
        
        # Add metrics plots on the right side
        if len(history['ph']) > 1:
            # Create a figure with subplots for metrics
            fig, axes = plt.subplots(4, 1, figsize=(4, 10))
            
            # Plot pH
            axes[0].plot(history['ph'], 'b-')
            axes[0].set_title('pH Level')
            axes[0].axhline(y=5.5, color='g', linestyle='--')
            axes[0].axhline(y=7.0, color='g', linestyle='--')
            
            # Plot EC
            axes[1].plot(history['ec'], 'r-')
            axes[1].set_title('EC Level')
            axes[1].axhline(y=1.0, color='g', linestyle='--')
            axes[1].axhline(y=3.0, color='g', linestyle='--')
            
            # Plot temperature
            axes[2].plot(history['temp'], 'orange')
            axes[2].set_title('Temperature')
            axes[2].axhline(y=18.0, color='g', linestyle='--')
            axes[2].axhline(y=26.0, color='g', linestyle='--')
            
            # Plot reward
            axes[3].plot(history['reward'], 'g-')
            axes[3].set_title('Reward')
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure to a buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            
            # Load the image from buffer
            img = imageio.imread(buf)
            buf.close()
            plt.close(fig)
            
            # Ensure the image has only 3 channels (RGB)
            if img.shape[2] == 4:  # If it has an alpha channel
                img = img[:, :, :3]  # Keep only RGB channels
            
            # Resize and place in the enhanced frame
            img_resized = cv2.resize(img, (380, 780))
            enhanced_frame[10:790, 810:1190] = img_resized
        
        # Add textual information
        cv2.putText(
            enhanced_frame,
            f"Step: {step_count} | Day: {info['total_days']} | Growth: {['Seedling', 'Vegetative', 'Flowering', 'Fruiting'][info['growth_stage']]}",
            (10, 630),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Add action information
        cv2.putText(
            enhanced_frame,
            f"Action: {action_meanings[action]}",
            (10, 670),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Add metrics
        cv2.putText(
            enhanced_frame,
            f"pH: {info['ph']:.2f} | EC: {info['ec']:.2f} | Temp: {info['temperature']:.2f}°C",
            (10, 710),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Add reward information
        cv2.putText(
            enhanced_frame,
            f"Reward: {reward:.2f} | Total: {total_reward:.2f}",
            (10, 750),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # Add to frames
        frames.append(enhanced_frame)
        
        # Increment step counter
        step_count += 1
        
        if done or truncated:
            break
    
    # Close environment
    env.close()
    
    # Save video
    if frames:
        print(f"Saving video with {len(frames)} frames at {fps} FPS...")
        try:
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Video saved to {video_path}")
        except Exception as e:
            print(f"Error saving video: {e}")
    else:
        print("No frames captured, video not saved.")

if __name__ == "__main__":
    # Example usage
    # create_requirements_file()
    # plot_learning_curve("models/dqn/logs")
    pass