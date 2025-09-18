import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

def plot_training_progress(episode_rewards, episode_lengths, save_path=None):
    """Plot training metrics: rewards and episode lengths over time"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot episode rewards
    ax1.plot(episode_rewards)
    ax1.set_title('Episode Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.grid(True)
    
    # Plot episode lengths
    ax2.plot(episode_lengths)
    ax2.set_title('Episode Lengths Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training plots saved to {save_path}")
    else:
        plt.show()

def save_model(policy_net, save_dir, episode, avg_reward):
    """Save trained model with metadata"""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"policy_net_ep{episode}_reward{avg_reward:.1f}_{timestamp}.pth"
    filepath = os.path.join(save_dir, filename)
    
    # Save model state dict and metadata
    torch.save({
        'model_state_dict': policy_net.state_dict(),
        'episode': episode,
        'avg_reward': avg_reward,
        'timestamp': timestamp,
        'model_architecture': {
            'input_size': 2,
            'hidden_size': 64,
            'output_size': 4
        }
    }, filepath)
    
    print(f"Model saved: {filepath}")
    return filepath

def load_model(policy_net, filepath):
    """Load trained model"""
    checkpoint = torch.load(filepath)
    policy_net.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {filepath}")
    print(f"Trained for {checkpoint['episode']} episodes")
    print(f"Average reward: {checkpoint['avg_reward']:.2f}")
    
    return checkpoint

def test_agent(env, policy_net, num_episodes=5, render=True):
    """Test trained agent performance"""
    total_rewards = []
    episode_lengths = []
    
    print(f"\n Testing agent for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(100):  # Max 100 steps per test episode
            # Use greedy action selection (no exploration)
            with torch.no_grad():
                action_probs = policy_net(state)
                action = torch.argmax(action_probs).item()
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render and episode == 0:  # Render first episode
                env.render()
            
            if done:
                break
        
        total_rewards.append(total_reward)
        episode_lengths.append(steps)

        status = "SUCCESS" if info.get('treasure_found', False) else "FAILED"
        print(f"Episode {episode+1}: {status} | Reward: {total_reward:.1f} | Steps: {steps}")
    
    avg_reward = np.mean(total_rewards)
    avg_length = np.mean(episode_lengths)
    success_rate = sum(1 for r in total_rewards if r > 0) / num_episodes
    
    print(f"\n Test Results:")
    print(f"Average Reward: {avg_reward:.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average Episode Length: {avg_length:.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Success Rate: {success_rate:.1%}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length,
        'success_rate': success_rate,
        'rewards': total_rewards,
        'lengths': episode_lengths
    }

def calculate_moving_average(data, window_size=100):
    """Calculate moving average for smoother plotting"""
    if len(data) < window_size:
        return data
    
    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size + 1)
        moving_avg.append(np.mean(data[start_idx:i+1]))
    
    return moving_avg

def print_training_stats(episode, episode_reward, episode_length, avg_reward_100, best_reward):
    """Print formatted training statistics"""
    print(f"Episode {episode:4d} | "
          f"Reward: {episode_reward:6.1f} | "
          f"Length: {episode_length:3d} | "
          f"Avg(100): {avg_reward_100:6.1f} | "
          f"Best: {best_reward:6.1f}")

def create_log_file(log_dir="logs"):
    """Create log file for training session"""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    with open(log_file, 'w') as f:
        f.write(f"REINFORCE Training Log - {timestamp}\n")
        f.write("=" * 50 + "\n")
        f.write("Episode,Reward,Length,Loss\n")  # CSV header
    
    return log_file

def log_episode(log_file, episode, reward, length, loss):
    """Log episode data to file"""
    with open(log_file, 'a') as f:
        f.write(f"{episode},{reward:.2f},{length},{loss:.6f}\n")

# Test the utils functions
if __name__ == "__main__":
    print("Testing utility functions...")

    
    
    # Test plotting with dummy data
    dummy_rewards = [i + np.random.randn() for i in range(100)]
    dummy_lengths = [50 - i//10 + np.random.randint(-5, 5) for i in range(100)]
    
    print(" Creating sample training plots...")
    plot_training_progress(dummy_rewards, dummy_lengths)
    
    # Test moving average
    moving_avg = calculate_moving_average(dummy_rewards, 10)
    print(f" Moving average calculated: {len(moving_avg)} points")
    
    # Test log file creation
    log_file = create_log_file()
    log_episode(log_file, 1, 10.5, 25, 0.123)
    print(f"Log file created: {log_file}")
    
    print("All utility functions working!")
