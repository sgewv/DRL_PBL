import gymnasium as gym
import random
import numpy as np
import torch

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that CUDA operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_environment(env_name, seed=None):
    """
    Creates a Gym environment and applies appropriate wrappers.
    """
    is_atari = "NoFrameskip" in env_name
    env = gym.make(env_name)
    
    if is_atari:
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
        env = gym.wrappers.FrameStack(env, 4)
    
    if seed is not None:
        set_seed(seed)
        
    return env