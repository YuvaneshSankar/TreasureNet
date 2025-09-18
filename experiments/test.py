import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time
from collections import defaultdict

# Add src directory to path so we can import our modules
sys.path.append('../src')

from policy_network import PolicyNetwork
from environment import GridWorldEnvironment
from reinforce_agent import ReinforceAgent

# Import configuration
sys.path.append('..')
import config

def load_trained_model(model_path, input_size=2, hidden_size=64, output_size=4):
    """Load a trained policy network from saved checkpoint"""
    print(f"Loading trained model from: {model_path}")
    
    # Load trained model
    policy_net = PolicyNetwork(
        input_size=input_size,
        hidden_size=hidden_size, 
        output_size=output_size
    )
    
    # Load saved model state
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path)
        if 'model_state_dict' in checkpoint:
            policy_net.load_state_dict(checkpoint['model_state_dict'])
            print(f" Model loaded successfully!")
            if 'episode' in checkpoint:
                print(f"   Trained for {checkpoint['episode']} episodes")
            if 'avg_reward' in checkpoint:
                print(f"   Final average reward: {checkpoint['avg_reward']:.2f}")
        else:
            # Direct state dict save
            policy_net.load_state_dict(checkpoint)
            print(f" Model state loaded successfully!")
    else:
        print(f" Model file not found: {model_path}")
        print(" Using randomly initialized model for testing...")
    
    return policy_net

def test_greedy_policy(env, policy_net, num_episodes=10, render=True, max_steps=50):
    """Test agent using greedy policy (no exploration)"""
    print(f"\n Testing Greedy Policy for {num_episodes} episodes...")
    
    test_results = {
        'rewards': [],
        'lengths': [],
        'successes': 0,
        'paths': [],
        'final_positions': []
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        path = [tuple(state.numpy())]
        
        if render and episode < 3:  # Render first 3 episodes
            print(f"\n🎮 Episode {episode + 1} Visualization:")
            env.render()
            time.sleep(0.5)
        
        for step in range(max_steps):
            # Greedy action selection (no exploration)
            with torch.no_grad():
                action_probs = policy_net(state)
                action = torch.argmax(action_probs).item()  # Best action
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            path.append(tuple(state.numpy()))
            
            if render and episode < 3:
                print(f"Step {step + 1}: Action {action} ({'up,down,left,right'.split(',')[action]})")
                env.render()
                time.sleep(0.3)
            
            if done:
                if info['treasure_found']:
                    test_results['successes'] += 1
                    if render and episode < 3:
                        print(" TREASURE FOUND!")
                elif info['max_steps_reached']:
                    if render and episode < 3:
                        print("Max steps reached")
                break
        
        test_results['rewards'].append(total_reward)
        test_results['lengths'].append(steps)
        test_results['paths'].append(path)
        test_results['final_positions'].append(tuple(state.numpy()))
        
        if render and episode < 3:
            print(f"Episode {episode + 1} Result: Reward = {total_reward:.1f}, Steps = {steps}")
            print("-" * 40)
    
    return test_results

def test_random_baseline(env, num_episodes=10):
    """Test random policy for comparison"""
    print(f"\n Testing Random Baseline for {num_episodes} episodes...")
    
    baseline_results = {
        'rewards': [],
        'lengths': [],
        'successes': 0
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(50):  # Max 50 steps
            action = np.random.randint(0, 4)  # Random action
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                if info['treasure_found']:
                    baseline_results['successes'] += 1
                break
        
        baseline_results['rewards'].append(total_reward)
        baseline_results['lengths'].append(steps)
    
    return baseline_results

def test_stochastic_policy(env, policy_net, num_episodes=10):
    """Test agent using stochastic policy (with exploration)"""
    print(f"\n Testing Stochastic Policy for {num_episodes} episodes...")
    
    stoch_results = {
        'rewards': [],
        'lengths': [],
        'successes': 0
    }
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        for step in range(50):  # Max 50 steps
            # Stochastic action selection (with exploration)
            with torch.no_grad():
                action_probs = policy_net(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample().item()  # Sample from distribution
            
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if done:
                if info['treasure_found']:
                    stoch_results['successes'] += 1
                break
        
        stoch_results['rewards'].append(total_reward)
        stoch_results['lengths'].append(steps)
    
    return stoch_results

def analyze_performance(greedy_results, random_results, stoch_results, env):
    """Analyze and compare performance across different policies"""
    print("\n" + "-" * 20)
    print("PERFORMANCE ANALYSIS")
    print("-" * 20)
    
    # Calculate metrics
    optimal_path_length = env.get_shortest_path_length()
    
    def calc_metrics(results, name):
        rewards = results['rewards']
        lengths = results['lengths']
        successes = results['successes']
        
        return {
            'name': name,
            'avg_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'avg_length': np.mean(lengths),
            'std_length': np.std(lengths),
            'success_rate': successes / len(rewards),
            'efficiency': np.mean([optimal_path_length / l if l > 0 else 0 for l in lengths])
        }
    
    greedy_metrics = calc_metrics(greedy_results, "Greedy Policy")
    random_metrics = calc_metrics(random_results, "Random Baseline")
    stoch_metrics = calc_metrics(stoch_results, "Stochastic Policy")
    
    # Print comparison table
    print(f"\n{'Policy':<18} | {'Avg Reward':<12} | {'Success Rate':<12} | {'Avg Length':<12} | {'Efficiency':<12}")
    print("-" * 80)
    for metrics in [greedy_metrics, stoch_metrics, random_metrics]:
        print(f"{metrics['name']:<18} | {metrics['avg_reward']:7.2f} ± {metrics['std_reward']:4.2f} | "
              f"{metrics['success_rate']:10.1%}   | {metrics['avg_length']:6.1f} ± {metrics['std_length']:4.1f} | "
              f"{metrics['efficiency']:10.2f}")
    
    print(f"\nOptimal path length: {optimal_path_length} steps")
    
    # Performance improvement
    reward_improvement = (greedy_metrics['avg_reward'] - random_metrics['avg_reward']) / abs(random_metrics['avg_reward']) * 100
    success_improvement = greedy_metrics['success_rate'] - random_metrics['success_rate']
    
    print(f"\n🚀 IMPROVEMENTS OVER RANDOM BASELINE:")
    print(f"   Reward improvement: {reward_improvement:+.1f}%")
    print(f"   Success rate improvement: {success_improvement:+.1%}")
    print(f"   Greedy vs Stochastic success rate: {greedy_metrics['success_rate']:.1%} vs {stoch_metrics['success_rate']:.1%}")
    
    return greedy_metrics, random_metrics, stoch_metrics

def plot_test_results(greedy_results, random_results, stoch_results):
    """Plot comprehensive test results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reward comparison
    policies = ['Greedy', 'Stochastic', 'Random']
    rewards = [greedy_results['rewards'], stoch_results['rewards'], random_results['rewards']]
    colors = ['green', 'blue', 'red']
    
    ax1.boxplot(rewards, labels=policies)
    ax1.set_title('Reward Distribution by Policy')
    ax1.set_ylabel('Episode Reward')
    ax1.grid(True, alpha=0.3)
    
    # Episode length comparison
    lengths = [greedy_results['lengths'], stoch_results['lengths'], random_results['lengths']]
    ax2.boxplot(lengths, labels=policies)
    ax2.set_title('Episode Length Distribution by Policy')
    ax2.set_ylabel('Steps to Complete/Timeout')
    ax2.grid(True, alpha=0.3)
    
    # Success rate comparison
    success_rates = [
        greedy_results['successes'] / len(greedy_results['rewards']),
        stoch_results['successes'] / len(stoch_results['rewards']),
        random_results['successes'] / len(random_results['rewards'])
    ]
    bars = ax3.bar(policies, success_rates, color=colors, alpha=0.7)
    ax3.set_title('Success Rate by Policy')
    ax3.set_ylabel('Success Rate')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, success_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    # Path visualization for first successful greedy episode
    successful_paths = [path for i, path in enumerate(greedy_results['paths']) 
                       if greedy_results['rewards'][i] > 0]
    
    if successful_paths:
        path = successful_paths[0]  # First successful path
        path_array = np.array(path)
        
        # Create grid background
        grid_size = 5
        ax4.set_xlim(-0.5, grid_size-0.5)
        ax4.set_ylim(-0.5, grid_size-0.5)
        
        # Plot obstacles (if available from environment)
        # obstacles = [(1, 3), (2, 1), (2, 2), (3, 2)]  # Example obstacles
        # for obs in obstacles:
        #     ax4.add_patch(plt.Rectangle((obs[1]-0.4, obs[0]-0.4), 0.8, 0.8, 
        #                                facecolor='red', alpha=0.5))
        
        # Plot path
        ax4.plot(path_array[:, 1], path_array[:, 0], 'bo-', linewidth=2, markersize=8, alpha=0.7)
        ax4.plot(path_array[0, 1], path_array[0, 0], 'go', markersize=15, label='Start')
        ax4.plot(path_array[-1, 1], path_array[-1, 0], 'ro', markersize=15, label='End')
        
        ax4.set_title('Sample Successful Path (Greedy Policy)')
        ax4.set_xlabel('Column')
        ax4.set_ylabel('Row')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
    else:
        ax4.text(0.5, 0.5, 'No successful\nepisodes found', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Path Visualization')
    
    plt.tight_layout()
    plt.savefig('test_results_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """Main testing function"""
    print("-" * 20)
    print("REINFORCE AGENT TESTING")
    print("-" * 20)
    
    # ================================
    # SETUP
    # ================================
    
    # Test parameters
    NUM_TEST_EPISODES = 20
    GRID_SIZE = 5
    MAX_STEPS = 50
    
    # Model path (modify this to your saved model path)
    MODEL_PATH = "saved_models/best_model.pth"  # Change this to your model path
    
    # Create environment
    env = GridWorldEnvironment(grid_size=GRID_SIZE, max_steps=MAX_STEPS)
    print(f"Test Environment: {GRID_SIZE}x{GRID_SIZE} grid")
    print(f"Start: {env.start_pos}, Treasure: {env.treasure_pos}")
    print(f"Obstacles: {env.obstacles}")
    
    # Load trained model
    policy_net = load_trained_model(MODEL_PATH)
    
    # ================================
    # RUN TESTS
    # ================================
    
    # Test 1: Greedy Policy (best performance)
    greedy_results = test_greedy_policy(env, policy_net, NUM_TEST_EPISODES, render=True)
    
    # Test 2: Stochastic Policy (with exploration)
    stoch_results = test_stochastic_policy(env, policy_net, NUM_TEST_EPISODES)
    
    # Test 3: Random Baseline (for comparison)
    random_results = test_random_baseline(env, NUM_TEST_EPISODES)
    
    # ================================
    # ANALYSIS & VISUALIZATION
    # ================================
    
    # Analyze performance
    analyze_performance(greedy_results, random_results, stoch_results, env)
    
    # Plot results
    plot_test_results(greedy_results, random_results, stoch_results)
    
    # ================================
    # DETAILED GREEDY POLICY ANALYSIS
    # ================================
    
    print(f"\nDETAILED GREEDY POLICY ANALYSIS:")
    greedy_rewards = greedy_results['rewards']
    greedy_lengths = greedy_results['lengths']
    
    print(f"   Best episode reward: {max(greedy_rewards):.1f}")
    print(f"   Worst episode reward: {min(greedy_rewards):.1f}")
    print(f"   Reward consistency (std/mean): {np.std(greedy_rewards)/abs(np.mean(greedy_rewards)):.2f}")
    print(f"   Shortest successful episode: {min([l for i, l in enumerate(greedy_lengths) if greedy_rewards[i] > 0], default='N/A')} steps")
    print(f"   Average successful episode: {np.mean([l for i, l in enumerate(greedy_lengths) if greedy_rewards[i] > 0]):.1f} steps" if greedy_results['successes'] > 0 else "   No successful episodes")
    
    return greedy_results, stoch_results, random_results

# ================================
# COMMAND LINE INTERFACE
# ================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test trained REINFORCE agent')
    parser.add_argument('--model', type=str, default='saved_models/best_model.pth',
                       help='Path to saved model file')
    parser.add_argument('--episodes', type=int, default=20,
                       help='Number of test episodes')
    parser.add_argument('--no-render', action='store_true',
                       help='Disable episode rendering')
    
    args = parser.parse_args()
    
    print(f"Testing with model: {args.model}")
    print(f"Number of episodes: {args.episodes}")
    print(f"Rendering: {'Disabled' if args.no_render else 'Enabled'}")
    
    try:
        # Run main testing
        greedy_results, stoch_results, random_results = main()
        
        print(f"\n Testing completed successfully!")
        print(f" Results saved to: test_results_analysis.png")
        
    except KeyboardInterrupt:
        print(f"\n Testing interrupted by user")
    except Exception as e:
        print(f"\n Testing failed with error: {e}")
        import traceback
        traceback.print_exc()
