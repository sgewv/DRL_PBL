import matplotlib.pyplot as plt
import torch
import os
import numpy as np

def plot_training_progress(episode_durations, episode_rewards, results_dir, episode_index, quiet=False):
    # Plot Episode Durations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.title(f'Episode Durations (Episode {episode_index})')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(episode_durations, label='Duration')

    if len(episode_durations) >= 100:
        moving_average_durations = torch.tensor(episode_durations).float().unfold(0, 100, 1).mean(1).view(-1)
        moving_average_durations = torch.cat((torch.zeros(99), moving_average_durations))
        plt.plot(moving_average_durations.numpy(), label='Moving Average (100)')
    plt.legend()

    # Plot Episode Rewards
    plt.subplot(1, 2, 2) # 1 row, 2 columns, 2nd plot
    plt.title(f'Episode Rewards (Episode {episode_index})')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards, label='Reward', color='green')

    if len(episode_rewards) >= 100:
        moving_average_rewards = torch.tensor(episode_rewards).float().unfold(0, 100, 1).mean(1).view(-1)
        moving_average_rewards = torch.cat((torch.zeros(99), moving_average_rewards))
        plt.plot(moving_average_rewards.numpy(), label='Moving Average (100)', color='red')
    plt.legend()
    
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    plt.savefig(os.path.join(results_dir, 'training_progress.png'))
    if not quiet:
        plt.show()
    plt.close() # Close the plot to free memory
