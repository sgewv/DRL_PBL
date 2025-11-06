import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import argparse
from itertools import count
from collections import deque
import numpy as np
import optuna
import os
from datetime import datetime

from src.agent import DQNAgent
from src.monitor import plot_training_progress
from src.utils import set_seed

def run_training(args, results_dir=None):
    """Runs the training loop and returns the final score."""
    env = gym.make(args.env_name)
    state, info = env.reset()
    state_dim = len(state)
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim, action_dim,
        use_dueling=args.use_dueling, use_double=args.use_double, use_per=args.use_per,
        use_noisy=args.use_noisy, use_distributional=args.use_distributional,
        num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max,
        learning_rate=args.lr
    )

    episode_durations = []
    episode_rewards = [] 
    n_step_buffer = deque(maxlen=args.n_steps)

    if not args.quiet:
        print(f"Starting training on {agent.device}...")
        print(f"Agent settings: Double: {agent.use_double}, Dueling: {agent.use_dueling}, PER: {agent.use_per}, Noisy: {agent.use_noisy}, N-Steps: {args.n_steps}, Distributional: {agent.use_distributional}")

    for episode_index in range(args.num_episodes):
        state, info = env.reset()
        agent.reset_noise()
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(0)
        n_step_buffer.clear()
        current_episode_reward = 0 # Initialize reward for current episode
        
        for time_step in count():
            action = agent.select_action(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            current_episode_reward += reward # Accumulate reward
            reward_tensor = torch.tensor([reward], device=agent.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=agent.device).unsqueeze(0)

            n_step_buffer.append((state, action, reward_tensor, next_state, done))

            if len(n_step_buffer) == args.n_steps:
                n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(args.n_steps)])
                sequence_initial_state, sequence_initial_action, _, _, _ = n_step_buffer[0]
                _, _, _, sequence_final_next_state, sequence_final_done_flag = n_step_buffer[-1]
                agent.add_to_memory(sequence_initial_state, sequence_initial_action, n_step_return, sequence_final_next_state, sequence_final_done_flag, args.gamma, args.n_steps)

            state = next_state

            agent.optimize_model(args.batch_size, args.gamma, args.n_steps)
            agent.update_target_net(args.tau)

            if done:
                while len(n_step_buffer) > 0:
                    n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(len(n_step_buffer))])
                    sequence_initial_state, sequence_initial_action, _, _, _ = n_step_buffer[0]
                    _, _, _, sequence_final_next_state, sequence_final_done_flag = n_step_buffer[-1]
                    agent.add_to_memory(sequence_initial_state, sequence_initial_action, n_step_return, sequence_final_next_state, sequence_final_done_flag, args.gamma, len(n_step_buffer))
                    n_step_buffer.popleft()

                episode_durations.append(time_step + 1)
                episode_rewards.append(current_episode_reward) # Append total reward for the episode
                if (episode_index + 1) % 10 == 0:
                    if not args.quiet:
                        print(f"Episode {episode_index+1}: duration {time_step+1}")
                    if results_dir:
                        plot_training_progress(episode_durations, episode_rewards, results_dir, episode_index + 1, args.quiet)
                
                if args.early_stop_threshold > 0 and len(episode_durations) >= 100:
                    avg_score = np.mean(episode_durations[-100:])
                    if avg_score >= args.early_stop_threshold:
                        if not args.quiet:
                            print(f"\nEarly stopping triggered at episode {episode_index+1} with average score {avg_score:.2f}")
                        break
                break
    
    if not args.quiet:
        print('Complete')
    env.close()

    score = -1
    if len(episode_durations) >= 100:
        score = np.mean(episode_durations[-100:])
    elif len(episode_durations) > 0:
        score = np.mean(episode_durations)

    return episode_durations, episode_rewards, score

def objective(trial, args):
    """Defines a single trial for Optuna to optimize."""
    trial_args = argparse.Namespace(**vars(args))

    # Tune hyperparameters
    trial_args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial_args.n_steps = trial.suggest_int("n_steps", 1, 5)
    
    # Set agent features based on search_mode
    if args.search_mode == 'all':
        trial_args.use_double = trial.suggest_categorical("use_double", [True, False])
        trial_args.use_dueling = trial.suggest_categorical("use_dueling", [True, False])
        trial_args.use_per = trial.suggest_categorical("use_per", [True, False])
        trial_args.use_noisy = trial.suggest_categorical("use_noisy", [True, False])
        trial_args.use_distributional = trial.suggest_categorical("use_distributional", [True, False])
    elif args.search_mode == 'rainbow':
        trial_args.use_double = True
        trial_args.use_dueling = True
        trial_args.use_per = True
        trial_args.use_noisy = True
        trial_args.use_distributional = True
    else:
        # Fix all to False, then enable the one specified by search_mode
        trial_args.use_double = False
        trial_args.use_dueling = False
        trial_args.use_per = False
        trial_args.use_noisy = False
        trial_args.use_distributional = False
        if args.search_mode != 'base':
            setattr(trial_args, f"use_{args.search_mode}", True)

    trial_args.quiet = True
    trial_args.num_episodes = args.num_episodes_per_trial
    trial_args.v_max = float(args.num_episodes_per_trial)

    param_str = f"lr={trial_args.lr:.6f}, n_steps={trial_args.n_steps}"
    print(f"\nStarting Trial {trial.number} ({args.search_mode} mode) with params: {param_str}")
    
    # Create results directory for this trial
    trial_results_dir = os.path.join("results", args.search_mode, f"trial_{trial.number}")
    os.makedirs(trial_results_dir, exist_ok=True)

    _, _, final_score = run_training(trial_args, trial_results_dir)

    print(f"Trial {trial.number} finished with score: {final_score:.2f}")

    return final_score

def start_optuna_search(args):
    """Starts the Optuna hyperparameter search."""
    set_seed(args.seed) # Set seed for reproducibility

    # Create results directory for Optuna search
    results_dir = os.path.join("results", args.search_mode)
    os.makedirs(results_dir, exist_ok=True)

    study = optuna.create_study(direction="maximize")
    objective_func = lambda trial: objective(trial, args)
    study.optimize(objective_func, n_trials=args.n_trials)

    print("\n--- Optimization Finished ---")
    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best trial for mode '{args.search_mode}':")
    trial = study.best_trial

    print(f"  Value (Score): {trial.value}")
    print("  Params: ")
    
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"best_trial_results_{timestamp}.txt"

    # Save best trial results to a file
    with open(os.path.join(results_dir, filename), "w") as f:
        f.write(f"Random Seed: {args.seed}\n")
        f.write(f"Best trial for mode '{args.search_mode}':\n")
        f.write(f"  Value (Score): {trial.value}\n")
        f.write("  Params: \n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")
            print(f"    {key}: {value}")

def main(args):
    """Runs a single training session and plots the result."""
    set_seed(args.seed) # Set seed for reproducibility

    # Create results directory if it doesn't exist
    results_dir = os.path.join("results", args.search_mode if args.search else "single_run")
    os.makedirs(results_dir, exist_ok=True)

    episode_durations, episode_rewards, final_score = run_training(args, results_dir)
    
    print(f"\nFinal Score (avg of last 100 episodes): {final_score:.2f}")

    # Plot Episode Durations
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1) # 1 row, 2 columns, 1st plot
    plt.title('Final Episode Durations')
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
    plt.title('Final Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episode_rewards, label='Reward', color='green')
    if len(episode_rewards) >= 100:
        moving_average_rewards = torch.tensor(episode_rewards).float().unfold(0, 100, 1).mean(1).view(-1)
        moving_average_rewards = torch.cat((torch.zeros(99), moving_average_rewards))
        plt.plot(moving_average_rewards.numpy(), label='Moving Average (100)', color='red')
    plt.legend()
    
    plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"training_result_{timestamp}.png"
    plt.savefig(os.path.join(results_dir, filename))
    if not args.quiet:
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DQN Agent Training and Optimization')

    parser.add_argument('--search', action='store_true', help='Run Optuna hyperparameter search')
    parser.add_argument('--search_mode', type=str, default='all',
                        choices=['all', 'base', 'double', 'dueling', 'per', 'noisy', 'distributional', 'rainbow'],
                        help='Mode for Optuna search. \'all\' searches combinations, others fix one feature.')

    # Agent feature flags
    parser.add_argument('--use_double', action='store_true', help='Enable Double DQN')
    parser.add_argument('--use_dueling', action='store_true', help='Enable Dueling Network')
    parser.add_argument('--use_per', action='store_true', help='Enable Prioritized Experience Replay')
    parser.add_argument('--use_noisy', action='store_true', help='Enable Noisy Nets')
    parser.add_argument('--use_distributional', action='store_true', help='Enable Distributional RL (C51)')
    parser.add_argument('--quiet', action='store_true', help='Suppress printouts and plot display for single runs')

    # Hyperparameters
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Gym environment name')
    parser.add_argument('--num_episodes', type=int, default=500, help='Number of episodes to train for a single run')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network')
    parser.add_argument('--n_steps', type=int, default=1, help='Number of steps for multi-step learning')
    parser.add_argument('--early_stop_threshold', type=int, default=495, help='Stop training if avg score of last 100 episodes reaches this threshold. Set to 0 to disable.')

    # Distributional RL parameters
    parser.add_argument('--num_atoms', type=int, default=51, help='Number of atoms for distributional RL')
    parser.add_argument('--v_min', type=float, default=-10.0, help='Minimum value of the value distribution')
    parser.add_argument('--v_max', type=float, default=10.0, help='Maximum value of the value distribution')

    # Optuna search parameters
    parser.add_argument('--n_trials', type=int, default=1, help='Number of trials for Optuna search')
    parser.add_argument('--num_episodes_per_trial', type=int, default=300, help='Number of episodes per Optuna trial')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')

    args = parser.parse_args()

    if args.search:
        start_optuna_search(args)
    else:
        main(args)
