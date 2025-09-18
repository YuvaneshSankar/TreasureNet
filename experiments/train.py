import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path so we can import our modules
sys.path.append('src')

from policy_network import PolicyNetwork
from environment import GridWorldEnvironment
from reinfroce_agent import ReinforceAgent

# Set random seeds for reproducibility (meaning everytime we get almost similar initial weights)
torch.manual_seed(42)  # Fix PyTorch random numbers
np.random.seed(42)     # Fix NumPy random numbers

def main():
    # ================================
    # HYPERPARAMETERS
    # ================================
    NUM_EPISODES = 1000          # Total training episodes
    GAMMA = 0.95                 # Discount factor for future rewards
    LEARNING_RATE = 0.001        # Neural network learning rate
    GRID_SIZE = 5                # Size of the grid world (5x5)
    MAX_STEPS = 50               # Maximum steps per episode
    PRINT_EVERY = 100            # Print progress every N episodes
    PLOT_EVERY = 200             # Plot training curves every N episodes
    TEST_EVERY = 200             # Test agent performance every N episodes
    
    print("🚀 Starting REINFORCE Training for Grid World Treasure Hunt")
    print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
    print(f"Training Episodes: {NUM_EPISODES}")
    print(f"Discount Factor (γ): {GAMMA}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("-" * 60)
    
    # ================================
    # INITIALIZE ENVIRONMENT & AGENT
    # ================================
    
    # Create the grid world environment
    env = GridWorldEnvironment(grid_size=GRID_SIZE, max_steps=MAX_STEPS)
    print(f"Environment created: {GRID_SIZE}x{GRID_SIZE} grid")
    print(f"Start: {env.start_pos}, Treasure: {env.treasure_pos}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Optimal path length: {env.get_shortest_path_length()}")
    
    # Create the policy network (2 inputs for x,y coordinates, 4 outputs for actions)
    policy_net = PolicyNetwork(
        input_size=2,           # x, y coordinates
        hidden_size=64,         # Hidden layer size
        output_size=4,          # 4 actions: up, down, left, right
        learning_rate=LEARNING_RATE
    )
    print(f"Policy network created: 2 -> 64 -> 64 -> 4")
    
    # Create the REINFORCE agent
    agent = ReinforceAgent(gamma=GAMMA)
    print(f"REINFORCE agent created with γ={GAMMA}")
    print("-" * 60)
    
    # ================================
    # TRAINING METRICS TRACKING
    # ================================
    episode_rewards = []        # Total reward per episode
    episode_lengths = []        # Number of steps per episode  
    episode_losses = []         # Policy loss per episode
    success_episodes = []       # Episodes where treasure was found
    
    best_reward = float('-inf') # Track best episode reward
    
    # ================================
    # MAIN TRAINING LOOP (SIMPLIFIED!)
    # ================================
    
    print(" Starting training...")
    
    for episode in range(1, NUM_EPISODES + 1):
        
        # STEP 1: Reset agent for new episode
        agent.reset()  # Clear previous episode data (log_probs, rewards)
        
        # STEP 2: Let agent handle entire episode collection
        episode_reward, episode_steps = agent.collect_episode(env, policy_net, MAX_STEPS)
        
        # STEP 3: Calculate policy loss and update network
        loss = agent.calculate_policy_loss()
        loss_value = policy_net.update_weights(loss)
        
        # STEP 4: Record episode metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        episode_losses.append(loss_value)
        
        # STEP 5: Check for success and update best reward
        if episode_reward > 0:  # Treasure found (positive reward)
            success_episodes.append(episode)
        
        if episode_reward > best_reward:
            best_reward = episode_reward
        
        # STEP 6: Print progress periodically
        if episode % PRINT_EVERY == 0:
            # Calculate recent performance
            recent_rewards = episode_rewards[-PRINT_EVERY:]
            avg_reward = np.mean(recent_rewards)
            success_rate = len([r for r in recent_rewards if r > 0]) / len(recent_rewards)
            avg_length = np.mean(episode_lengths[-PRINT_EVERY:])
            
            print(f"Episode {episode:4d} | "
                  f"Avg Reward: {avg_reward:6.1f} | "
                  f"Avg Length: {avg_length:4.1f} | "
                  f"Success Rate: {success_rate:4.1%} | "
                  f"Best: {best_reward:6.1f}")
        
        # STEP 7: Plot training curves periodically
        if episode % PLOT_EVERY == 0:
            plot_training_progress(episode_rewards, episode_lengths, episode)
        
        # STEP 8: Test agent performance periodically
        if episode % TEST_EVERY == 0:
            test_agent_performance(env, policy_net, episode)
    
    # ================================
    # TRAINING COMPLETED
    # ================================
    
    print("\n" + "-" * 20)
    print("TRAINING COMPLETED!")
    print("-" * 20)
    
    # Final performance summary
    final_100_rewards = episode_rewards[-100:]
    final_avg_reward = np.mean(final_100_rewards)
    final_success_rate = len([r for r in final_100_rewards if r > 0]) / len(final_100_rewards)
    total_successes = len(success_episodes)
    
    print(f"\n FINAL PERFORMANCE:")
    print(f"Final 100 episodes average reward: {final_avg_reward:.2f}")
    print(f"Final 100 episodes success rate: {final_success_rate:.1%}")
    print(f"Total successful episodes: {total_successes}/{NUM_EPISODES}")
    print(f"Best episode reward: {best_reward:.1f}")
    
    # Plot final training curves
    plot_final_results(episode_rewards, episode_lengths, success_episodes)
    
    # Final agent test
    print("\n🧪 FINAL AGENT TESTING:")
    test_agent_performance(env, policy_net, NUM_EPISODES, num_test_episodes=10, render=True)
    
    return policy_net, episode_rewards, episode_lengths, success_episodes

def plot_training_progress(rewards, lengths, episode):
    """Plot training progress during training"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot episode rewards with moving average
    ax1.plot(rewards, alpha=0.6, color='blue', label='Episode Reward')
    if len(rewards) >= 50:
        moving_avg = [np.mean(rewards[max(0, i-49):i+1]) for i in range(len(rewards))]
        ax1.plot(moving_avg, color='red', label='Moving Average (50)')
    ax1.set_title(f'Episode Rewards (Episode {episode})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(lengths, alpha=0.6, color='green', label='Episode Length')
    if len(lengths) >= 50:
        moving_avg = [np.mean(lengths[max(0, i-49):i+1]) for i in range(len(lengths))]
        ax2.plot(moving_avg, color='red', label='Moving Average (50)')
    ax2.set_title(f'Episode Lengths (Episode {episode})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_progress_ep{episode}.png')
    plt.show()

def plot_final_results(rewards, lengths, success_episodes):
    """Plot comprehensive final results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Rewards over time
    ax1.plot(rewards, alpha=0.7, color='blue', label='Episode Reward')
    moving_avg = [np.mean(rewards[max(0, i-99):i+1]) for i in range(len(rewards))]
    ax1.plot(moving_avg, color='red', linewidth=2, label='Moving Average (100)')
    ax1.set_title('Training Rewards Over Time')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Episode lengths over time
    ax2.plot(lengths, alpha=0.7, color='green', label='Episode Length')
    moving_avg = [np.mean(lengths[max(0, i-99):i+1]) for i in range(len(lengths))]
    ax2.plot(moving_avg, color='red', linewidth=2, label='Moving Average (100)')
    ax2.set_title('Episode Lengths Over Time')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps to Complete')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Success rate over time
    success_rate = []
    window_size = 100
    for i in range(len(rewards)):
        start_idx = max(0, i - window_size + 1)
        window_rewards = rewards[start_idx:i+1]
        rate = sum(1 for r in window_rewards if r > 0) / len(window_rewards)
        success_rate.append(rate * 100)
    
    ax3.plot(success_rate, color='purple', linewidth=2)
    ax3.set_title(f'Success Rate Over Time (Window: {window_size})')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Success Rate (%)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Reward distribution histogram
    ax4.hist(rewards, bins=30, alpha=0.7, color='orange', edgecolor='black')
    ax4.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.1f}')
    ax4.set_title('Distribution of Episode Rewards')
    ax4.set_xlabel('Total Reward')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_training_results.png', dpi=150)
    plt.show()

def test_agent_performance(env, policy_net, current_episode, num_test_episodes=5, render=False):
    """Test the agent's current performance without exploration"""
    print(f"\nTesting agent at episode {current_episode}...")
    
    test_rewards = []
    test_lengths = []
    successes = 0
    
    for test_ep in range(num_test_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(50):  # Max 50 steps for testing
            # Use greedy action selection (no exploration)
            with torch.no_grad():
                action_probs = policy_net(state)
                action = torch.argmax(action_probs).item()  # Choose best action
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render and test_ep == 0:
                env.render()  # Show first test episode
            
            if done:
                if info['treasure_found']:
                    successes += 1
                break
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
    
    # Print test results
    avg_reward = np.mean(test_rewards)
    avg_length = np.mean(test_lengths)
    success_rate = successes / num_test_episodes
    
    print(f"   Test Results: Avg Reward: {avg_reward:.1f}, "
          f"Avg Length: {avg_length:.1f}, Success Rate: {success_rate:.1%}")
    
    return {
        'avg_reward': avg_reward,
        'avg_length': avg_length, 
        'success_rate': success_rate
    }

# ================================
# RUN TRAINING
# ================================

if __name__ == "__main__":
    print("Starting REINFORCE Grid World Training...")
    
    try:
        # Run the main training function
        trained_policy, rewards, lengths, successes = main()
        
        print(f"\n Training completed successfully!")
        print(f" Trained policy network ready for deployment!")
        
    except KeyboardInterrupt:
        print(f"\nTraining interrupted by user")
    except Exception as e:
        print(f"\n Training failed with error: {e}")
        import traceback
        traceback.print_exc()
