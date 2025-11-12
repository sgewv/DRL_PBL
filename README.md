# A Flexible Framework for Deep Reinforcement Learning

A modular PyTorch framework for Deep Reinforcement Learning, designed as a comprehensive teaching example. It features a custom session-based movie recommendation environment (`MovieRec-v1`), supports standard Gymnasium tasks, and integrates a suite of advanced DRL algorithms from the Rainbow paper.

## Key Features

- **Modular Design:** A clean separation of concerns, with distinct modules for the agent, network models, training, evaluation, and configuration.
- **Extensible:** Easily add new algorithms, models, or custom environments.
- **Advanced Algorithms:** Implements 6 key components from Rainbow, all toggleable via command-line flags.
- **Hyperparameter Tuning:** Integrated with Optuna for automated hyperparameter search.
- **Experiment Tracking:** Full support for Weights & Biases (W&B) to log metrics, models, and configurations.
- **Qualitative Analysis:** Includes a dedicated evaluation mode to observe a trained agent's behavior step-by-step.

## Implemented Algorithms Explained

This framework implements the core components of Rainbow, allowing you to combine them flexibly.

- **Double DQN (`--use_double`)**
  Reduces Q-value overestimation by decoupling action selection (using the policy network) from action evaluation (using the target network), leading to more stable training.

- **Dueling Networks (`--use_dueling`)**
  Improves learning efficiency by using a network architecture that separates the estimation of the state value (`V(s)`) from the advantage of each action (`A(s, a)`).

- **Prioritized Experience Replay (PER) (`--use_per`)**
  Accelerates learning by replaying "important" transitions—where the agent's prediction was very wrong (high TD-error)—more frequently.

- **N-Step Learning (`--n_steps [k]`)**
  Speeds up learning by calculating the TD target using the cumulative reward over `N` future steps, rather than just the single next step.

- **Distributional RL (C51) (`--use_distributional`)**
  Models the full distribution of expected returns instead of just the mean (the Q-value). This captures uncertainty and often leads to more stable and effective learning.

- **Noisy Nets (`--use_noisy`)**
  Provides a more sophisticated method for exploration by adding parametric noise directly to the network's weights, which can be learned and adapted during training.

## Project Structure

The project is organized into a modular structure to promote clarity and separation of concerns.

```
/
├── main.py                     # Main entry point to run training, evaluation, or hyperparameter search.
├── requirements.txt            # Project dependencies.
├── README.md                   # This file.
├── .gitignore                  # Specifies files and directories to be ignored by Git.
│
├── src/
│   ├── __init__.py
│   ├── agent.py                # Contains the DQNAgent class, implementing all algorithm logic.
│   ├── models.py               # Defines all neural network architectures (MLP, CNN, Dueling, Noisy).
│   ├── config.py               # Centralizes all command-line argument parsing (`argparse`).
│   ├── trainer.py              # Manages the entire training lifecycle.
│   ├── evaluator.py            # Manages the evaluation of trained agents for qualitative analysis.
│   ├── hyperparameter_tuning.py # Orchestrates the Optuna hyperparameter search.
│   ├── replay_buffer.py        # Implements both standard and Prioritized Experience Replay (PER) with a SumTree.
│   ├── utils.py                # Helper functions, including `set_seed` and the `create_environment` factory.
│   │
│   └── custom_envs/
│       ├── __init__.py         # Registers the custom environment with Gymnasium.
│       └── movie_rec.py        # Implementation of the custom `MovieRec-v1` environment.
│
├── results/                    # (Git-ignored) Stores training artifacts.
│   └── models/                 # Stores the best-performing model weights (`.pth` files).
│
└── wandb/                      # (Git-ignored) Directory for local Weights & Biases logs.
```

## Usage

### 1. Installation
It is recommended to use a virtual environment (e.g., Conda).

```bash
pip install -r requirements.txt
```

### 2. Training (학습)
Train an agent on a specified environment. All results are logged to W&B.

```bash
python main.py --env_name CartPole-v1

python main.py --env_name CartPole-v1 --use_double

python main.py --env_name CartPole-v1 --use_dueling

python main.py --env_name CartPole-v1 --use_per

python main.py --env_name CartPole-v1 --use_noisy

python main.py --env_name CartPole-v1 --use_distributional

python main.py --env_name CartPole-v1 --n_steps 3

python main.py --env_name CartPole-v1 --use_double --use_dueling

python main.py --env_name CartPole-v1 --use_per --n_steps 3

python main.py --env_name CartPole-v1 --use_double --use_dueling --use_per --use_distributional --n_steps 3

python main.py --env_name CartPole-v1 --use_double --use_dueling --use_per --use_noisy --use_distributional --n_steps 3

python main.py --env_name CartPole-v1 --lr 1e-5 --batch_size 256

python main.py --env_name CartPole-v1 --gamma 0.995 --tau 0.01

python main.py --env_name CartPole-v1 --eps_start 1.0 --eps_end 0.01 --eps_decay 2000

python main.py --env_name MovieRec-v1 --num_episodes 5000 --use_double --use_dueling

python main.py --env_name "ALE/Pong-v5" --num_episodes 20000 --use_per --n_steps 3

python main.py --evaluate --env_name CartPole-v1 --load_model_path "path/to/model.pth"

python main.py --evaluate --env_name CartPole-v1 --load_model_path "path/to/dueling_model.pth" --use_dueling

python main.py --evaluate --env_name CartPole-v1 --load_model_path "path/to/dist_model.pth" --use_distributional

python main.py --evaluate --env_name CartPole-v1 --load_model_path "path/to/model.pth" --eval_episodes 50

python main.py --search --env_name CartPole-v1 --n_trials 100

python main.py --search --env_name CartPole-v1 --n_trials 200 --search_mode all

python main.py --search --env_name CartPole-v1 --n_trials 100 --num_episodes_per_trial 500

python main.py --env_name CartPole-v1 --seed 123

python main.py --env_name CartPole-v1 --wandb_disable
```

### 3. Evaluation (평가)
Load a trained model to observe its behavior without exploration.

**Step A: Train a model and save it**
The training script automatically saves the model with the best average reward to `results/models/`.

```bash
# This will save a file like 'results/models/best_model_[run_id].pth'
python main.py --env_name MovieRec-v1 --num_episodes 1000
```

**Step B: Run the evaluation mode with the saved model**
Replace the path with your saved model's path.

```bash
python main.py --evaluate --load_model_path "results/models/best_model_... .pth" --env_name MovieRec-v1
```

### 4. Hyperparameter Search (하이퍼파라미터 탐색)
Use Optuna to automatically find optimal hyperparameters.

**Example: Run 50 trials on `CartPole-v1`**
```bash
python main.py --search --env_name CartPole-v1 --n_trials 50
```

## Command-Line Arguments

| Category | Argument | Default | Description |
|---|---|---|---|
| **Main** | `--env_name` | `CartPole-v1` | Name of the Gymnasium environment to run. |
| | `--num_episodes` | `2000` | Total number of episodes for training. |
| | `--seed` | `42` | Random seed for reproducibility. |
| | `--quiet` | `False` | Suppress detailed printouts during training. |
| **Hyperparameters** | `--lr` | `1e-4` | Learning rate for the AdamW optimizer. |
| | `--gamma` | `0.99` | Discount factor for future rewards. |
| | `--batch_size` | `128` | Number of transitions to sample from the replay buffer. |
| | `--tau` | `0.005` | Coefficient for soft updating the target network. |
| | `--n_steps` | `1` | Number of steps for N-step learning. |
| **Epsilon-Greedy** | `--eps_start` | `0.9` | The starting value of epsilon. |
| | `--eps_end` | `0.05` | The final value of epsilon. |
| | `--eps_decay` | `1000` | The decay rate of epsilon. |
| **Agent Features** | `--use_double` | `False` | Enable Double DQN. |
| | `--use_dueling` | `False` | Enable Dueling Network architecture. |
| | `--use_per` | `False` | Enable Prioritized Experience Replay. |
| | `--use_noisy` | `False` | Enable Noisy Nets for exploration. |
| | `--use_distributional`| `False` | Enable Distributional DQN (C51). |
| | `--num_atoms` | `51` | Number of atoms for Distributional DQN. |
| | `--v_min` | `-10.0` | Minimum value of the Q-value distribution. |
| | `--v_max` | `10.0` | Maximum value of the Q-value distribution. |
| **Evaluation** | `--evaluate` | `False` | Run in evaluation mode. |
| | `--load_model_path` | `None` | Path to the saved model for evaluation. |
| | `--eval_episodes` | `10` | Number of episodes to run for evaluation. |
| **Hyperparameter Search** | `--search` | `False` | Run Optuna hyperparameter search. |
| | `--n_trials` | `100` | Number of Optuna trials to run. |
| | `--num_episodes_per_trial`| `300` | Number of episodes per Optuna trial. |
| | `--search_mode` | `base` | Hyperparameter set to search (`base` or `all`). |
| **W&B Logging** | `--wandb_project` | `drl-pbl-lecture` | Name of the Weights & Biases project. |
| | `--wandb_disable` | `False` | Disable Weights & Biases logging. |


## References
- **Human-level control through deep reinforcement learning** (DQN)
  - Mnih, V., et al. (2015). *Nature*.
- **Playing atari with deep reinforcement learning** (DQN for Atari)
  - Mnih, V., et al. (2013). *arXiv*.
- **Rainbow: Combining improvements in deep reinforcement learning** (The paper that combines all the techniques below)
  - Hessel, M., et al. (2018). *AAAI*.
- **Noisy Nets for Exploration** (Noisy Nets)
  - Fortunato, M., et al. (2017). *arXiv*.
- **A Distributional Perspective on Reinforcement Learning** (Distributional RL, C51)
  - Bellemare, M. G., et al. (2017). *ICML*.
- **Deep reinforcement learning with double q-learning** (Double DQN)
  - Van Hasselt, H., et al. (2016). *AAAI*.
- **Dueling network architectures for deep reinforcement learning** (Dueling Networks)
  - Wang, Z., et al. (2016). *ICML*.
- **Prioritized experience replay** (PER)
  - Schaul, T., et al. (2015). *arXiv*.

## Contact
iudr42@gmail.com
