import torch
import numpy as np
import random

class GridWorldEnvironment:
    def __init__(self, grid_size=5, max_steps=100):
        self.grid_size = grid_size
        self.max_steps = max_steps
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.actions = {
            0: (-1, 0),  # up (decrease row)
            1: (1, 0),   # down (increase row) 
            2: (0, -1),  # left (decrease col)
            3: (0, 1)    # right (increase col)
        }
        
        # Initialize positions
        self.start_pos = (0, 0)  # Top-left corner
        self.treasure_pos = (grid_size-1, grid_size-1)  # Bottom-right corner
        
        # Add some obstacles (optional - you can remove these)
        self.obstacles = self._create_obstacles()
        
        # Current state
        self.current_pos = None
        self.steps_taken = 0
        self.done = False

        
        
    def _create_obstacles(self):
        """Create a few obstacles to make it more interesting"""
        obstacles = set()
        
        # Add some obstacles in the middle (avoid start and treasure)
        if self.grid_size >= 5:
            obstacles.add((2, 1))
            obstacles.add((2, 2))
            obstacles.add((1, 3))
            obstacles.add((3, 2))
        
        return obstacles
    
    def reset(self):
        """Reset environment to starting state"""
        self.current_pos = self.start_pos
        self.steps_taken = 0
        self.done = False
        return self.get_state()
    
    def get_state(self):
        """Return current state as tensor [x, y]"""
        return torch.tensor([float(self.current_pos[0]), float(self.current_pos[1])])
    
    def step(self, action):
        """Execute action and return (next_state, reward, done, info)"""
        if self.done:
            return self.get_state(), 0.0, True, {"message": "Episode already finished"}
        
        # Get action deltas
        if action not in self.actions:
            return self.get_state(), -1.0, False, {"message": "Invalid action"}
        
        delta_row, delta_col = self.actions[action]
        new_row = self.current_pos[0] + delta_row
        new_col = self.current_pos[1] + delta_col
        new_pos = (new_row, new_col)
        
        # Calculate reward
        reward = self._calculate_reward(new_pos)
        
        # Update position (only if valid move)
        if self._is_valid_position(new_pos):
            self.current_pos = new_pos
        
        # Update step counter
        self.steps_taken += 1
        
        # Check if episode is done
        self.done = self._is_episode_done()
        
        info = {
            "steps_taken": self.steps_taken,
            "treasure_found": self.current_pos == self.treasure_pos,
            "max_steps_reached": self.steps_taken >= self.max_steps
        }
        
        return self.get_state(), reward, self.done, info
    
    def _is_valid_position(self, pos):
        """Check if position is within bounds and not an obstacle"""
        row, col = pos
        
        # Check boundaries
        if row < 0 or row >= self.grid_size or col < 0 or col >= self.grid_size:
            return False
        
        # Check obstacles
        if pos in self.obstacles:
            return False
        
        return True
    
    def _calculate_reward(self, new_pos):
        """Calculate reward for moving to new_pos"""
        # Hit wall or obstacle - negative reward, don't move
        if not self._is_valid_position(new_pos):
            return -5.0
        
        # Reached treasure - big positive reward
        if new_pos == self.treasure_pos:
            return +10.0
        
        # Normal step - small negative reward (encourage efficiency)
        return -1.0
    
    def _is_episode_done(self):
        """Check if episode should end"""
        # Found treasure
        if self.current_pos == self.treasure_pos:
            return True
        
        # Max steps reached
        if self.steps_taken >= self.max_steps:
            return True
        
        return False
    
    def render(self):
        """Print current grid state (for debugging)"""
        print(f"\nStep {self.steps_taken}/{self.max_steps}")
        print("Grid (A=Agent, T=Treasure, X=Obstacle, .=Empty):")
        
        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                pos = (row, col)
                if pos == self.current_pos:
                    line += "A "
                elif pos == self.treasure_pos:
                    line += "T "
                elif pos in self.obstacles:
                    line += "X "
                else:
                    line += ". "
            print(line)
        print()
    
    def get_shortest_path_length(self):
        """Calculate shortest path length (for comparison)"""
        # Simple Manhattan distance ignoring obstacles
        return abs(self.treasure_pos[0] - self.start_pos[0]) + abs(self.treasure_pos[1] - self.start_pos[1])

# Testing code
if __name__ == "__main__":
    print("Testing GridWorld Environment...")
    
    # Create environment
    env = GridWorldEnvironment(grid_size=5, max_steps=50)
    
    print(f"Grid size: {env.grid_size}x{env.grid_size}")
    print(f"Start position: {env.start_pos}")
    print(f"Treasure position: {env.treasure_pos}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Shortest path length: {env.get_shortest_path_length()}")
    print()
    
    # Test reset
    print("Test 1: Reset")
    state = env.reset()
    print(f"Initial state: {state}")
    print(f"State type: {type(state)}")
    print(f"State shape: {state.shape}")
    env.render()
    
    # Test valid moves
    print("Test 2: Valid moves")
    actions = [1, 3, 1, 3, 1, 3]  # down, right, down, right, down, right
    
    for i, action in enumerate(actions):
        print(f"Step {i+1}: Action {action} ({'up,down,left,right'.split(',')[action]})")
        state, reward, done, info = env.step(action)
        print(f"  State: {state}, Reward: {reward}, Done: {done}")
        print(f"  Info: {info}")
        env.render()
        
        if done:
            print("Episode finished!")
            break
    
    # Test invalid moves (hitting walls)
    print("Test 3: Invalid moves (walls)")
    env.reset()
    
    # Try moving up from start (should hit wall)
    print("Trying to move up from (0,0)...")
    state, reward, done, info = env.step(0)  # up
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    
    # Try moving left from start (should hit wall)  
    print("Trying to move left from (0,0)...")
    state, reward, done, info = env.step(2)  # left
    print(f"State: {state}, Reward: {reward}, Done: {done}")
    
    # Test random episode
    print("\nTest 4: Random episode")
    env.reset()
    total_reward = 0
    
    for step in range(20):
        action = random.randint(0, 3)
        state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step+1}: Action {action}, Reward {reward}, Total: {total_reward}")
        
        if done:
            print(f"Episode finished! Total reward: {total_reward}")
            print(f"Reason: {'Treasure found!' if info['treasure_found'] else 'Max steps reached'}")
            break
    
    print("\n Environment testing complete!")
