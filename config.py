"""
Configuration file for REINFORCE Grid World Training
Contains all hyperparameters and settings in one centralized location
"""

# ================================
# TRAINING HYPERPARAMETERS
# ================================

# Episode and training settings
NUM_EPISODES = 1000          # Total number of training episodes
GAMMA = 0.95                 # Discount factor for future rewards
LEARNING_RATE = 0.001        # Neural network learning rate
MAX_STEPS = 50               # Maximum steps per episode

# Environment settings
GRID_SIZE = 5                # Size of the grid world (5x5)

# Network architecture
INPUT_SIZE = 2               # State representation: (x, y) coordinates
HIDDEN_SIZE = 64             # Hidden layer size
OUTPUT_SIZE = 4              # Number of actions: up, down, left, right

# ================================
# LOGGING AND VISUALIZATION
# ================================

# Progress reporting intervals
PRINT_EVERY = 100            # Print training progress every N episodes
PLOT_EVERY = 200             # Plot training curves every N episodes
TEST_EVERY = 200             # Test agent performance every N episodes

# File paths
LOG_DIR = "logs"             # Directory for training logs
SAVE_DIR = "saved_models"    # Directory for model checkpoints
PLOT_DIR = "plots"           # Directory for training plots

# ================================
# REPRODUCIBILITY
# ================================

# Random seeds for consistent results
PYTORCH_SEED = 42            # PyTorch random seed
NUMPY_SEED = 42              # NumPy random seed

# ================================
# TESTING PARAMETERS
# ================================

# Testing settings
TEST_EPISODES = 20           # Number of episodes for testing
TEST_MAX_STEPS = 50          # Maximum steps per test episode
RENDER_TEST = True           # Whether to render test episodes

# ================================
# MODEL SAVING
# ================================

# Model checkpoint settings
SAVE_BEST_MODEL = True       # Save model with best performance
SAVE_INTERVAL = 500          # Save checkpoint every N episodes
MODEL_NAME_PREFIX = "reinforce_gridworld"

# Performance thresholds
MIN_SUCCESS_RATE = 0.8       # Minimum success rate to save model
MIN_EPISODES_FOR_SAVE = 100  # Minimum episodes before saving

# ================================
# ADVANCED SETTINGS
# ================================

# Training optimizations
USE_BASELINE = True          # Use baseline for variance reduction
NORMALIZE_REWARDS = True     # Normalize rewards during training
CLIP_GRADIENTS = False       # Whether to clip gradients
MAX_GRAD_NORM = 1.0          # Maximum gradient norm if clipping

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Auto-detect device

# ================================
# ENVIRONMENT CUSTOMIZATION
# ================================

# Grid world specific settings
START_POSITION = (0, 0)              # Starting position (top-left)
TREASURE_POSITION = (4, 4)           # Treasure position (bottom-right)
CUSTOM_OBSTACLES = [                 # Custom obstacle positions
    (1, 3), (2, 1), (2, 2), (3, 2)
]

# Reward structure
TREASURE_REWARD = 10.0       # Reward for finding treasure
STEP_PENALTY = -1.0          # Penalty per step (encourages efficiency)
WALL_PENALTY = -5.0          # Penalty for hitting walls/obstacles

# ================================
# DEBUGGING AND DEVELOPMENT
# ================================

# Debug settings
DEBUG_MODE = False           # Enable debug prints
VERBOSE_LOGGING = False      # Detailed logging
SAVE_EPISODE_DATA = False    # Save individual episode data

# Quick test settings (for development)
QUICK_TEST_EPISODES = 10     # Reduced episodes for quick testing
QUICK_TEST_MAX_STEPS = 20    # Reduced steps for quick testing
