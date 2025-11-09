import argparse

def get_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description='DQN Agent Training Framework')

    # --- Main Arguments ---
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='Gym environment name')
    parser.add_argument('--num_episodes', type=int, default=2000, help='Number of episodes for a single run')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--quiet', action='store_true', help='Suppress printouts')

    # --- Hyperparameters ---
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update coefficient for target network')
    parser.add_argument('--n_steps', type=int, default=1, help='N-step learning')

    # --- Epsilon-Greedy Hyperparameters ---
    parser.add_argument('--eps_start', type=float, default=0.9, help='Epsilon starting value')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Epsilon ending value')
    parser.add_argument('--eps_decay', type=int, default=1000, help='Epsilon decay rate')

    # --- Agent Feature Flags ---
    parser.add_argument('--use_double', action='store_true', help='Enable Double DQN')
    parser.add_argument('--use_dueling', action='store_true', help='Enable Dueling Network')
    parser.add_argument('--use_per', action='store_true', help='Enable Prioritized Experience Replay')
    parser.add_argument('--use_noisy', action='store_true', help='Enable Noisy Nets')
    parser.add_argument('--use_distributional', action='store_true', help='Use Distributional DQN')
    parser.add_argument('--num_atoms', type=int, default=51, help='Number of atoms for Distributional DQN')
    parser.add_argument('--v_min', type=float, default=-10.0, help='Minimum value of support for Distributional DQN')
    parser.add_argument('--v_max', type=float, default=10.0, help='Maximum value of support for Distributional DQN')

    # --- Optuna Search ---
    parser.add_argument('--search', action='store_true', help='Run Optuna hyperparameter search')
    parser.add_argument('--n_trials', type=int, default=100, help='Number of Optuna trials')
    parser.add_argument('--num_episodes_per_trial', type=int, default=300, help='Number of episodes per Optuna trial')
    parser.add_argument('--search_mode', type=str, default='base', choices=['base', 'all'], help='Which hyperparameters to search')

    # --- W&B Logging ---
    parser.add_argument('--wandb_project', type=str, default='drl-pbl-lecture', help='W&B project name')
    parser.add_argument('--wandb_disable', action='store_true', help='Disable W&B logging')

    # --- Evaluation Arguments ---
    parser.add_argument('--evaluate', action='store_true', help='Run in evaluation mode')
    parser.add_argument('--load_model_path', type=str, default=None, help='Path to the model to load for evaluation')
    parser.add_argument('--eval_episodes', type=int, default=10, help='Number of episodes to run for evaluation')

    return parser.parse_args()
