import gymnasium as gym
import torch
import numpy as np
from itertools import count
from collections import deque
from datetime import datetime
from pathlib import Path
import wandb

# Register our custom environments
import src.custom_envs

from .agent import DQNAgent
from .utils import set_seed, create_environment

def run_training(args, trial=None):
    """Runs the main training loop."""
    # --- 1. Setup ---
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        name=f"{args.env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        reinit=True,
        mode="disabled" if args.wandb_disable else "online"
    )
    
    is_atari = "NoFrameskip" in args.env_name
    env = create_environment(args.env_name, args.seed)
    set_seed(args.seed)

    state_dim = env.observation_space.shape if is_atari else env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim, action_dim, is_atari,
        use_dueling=args.use_dueling, use_double=args.use_double,
        use_per=args.use_per, use_noisy=args.use_noisy,
        use_distributional=args.use_distributional, num_atoms=args.num_atoms,
        v_min=args.v_min, v_max=args.v_max,
        learning_rate=args.lr
    )

    # --- 2. Model Saving Setup ---
    model_dir = Path("results/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = model_dir / f"best_model_{wandb.run.id}.pth"
    best_avg_reward = -float('inf')

    if not args.quiet:
        print(f"Starting training on {args.env_name} | Device: {agent.device}")
        print(f"Config: {vars(args)}")
    
    # --- 3. Training Loop ---
    total_rewards = []
    n_step_buffer = deque(maxlen=args.n_steps)

    # Adjust epsilon schedule for Atari
    eps_start = 1.0 if is_atari else args.eps_start
    eps_end = 0.1 if is_atari else args.eps_end
    eps_decay = 1000000 if is_atari else args.eps_decay

    for episode_index in range(args.num_episodes):
        state, info = env.reset(seed=args.seed + episode_index)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
        
        current_episode_reward = 0
        
        for time_step in count():
            action = agent.select_action(state, eps_start, eps_end, eps_decay)
            observation, reward, terminated, truncated, info = env.step(action.item())
            
            current_episode_reward += reward
            reward_tensor = torch.tensor([reward], device=agent.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(agent.device)

            n_step_buffer.append((state, action, reward_tensor, next_state, done))

            if len(n_step_buffer) == args.n_steps:
                n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(args.n_steps)])
                start_state, start_action, _, _, _ = n_step_buffer[0]
                _, _, _, end_next_state, end_done = n_step_buffer[-1]
                agent.add_to_memory(start_state, start_action, n_step_return, end_next_state, end_done)

            state = next_state

            loss = agent.optimize_model(args.batch_size, args.gamma, args.n_steps)
            if loss is not None:
                wandb.log({"loss": loss}, step=agent.steps_done)

            agent.update_target_net(args.tau)

            if done:
                while len(n_step_buffer) > 0:
                    n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(len(n_step_buffer))])
                    start_state, start_action, _, _, _ = n_step_buffer[0]
                    _, _, _, end_next_state, end_done = n_step_buffer[-1]
                    agent.add_to_memory(start_state, start_action, n_step_return, end_next_state, end_done)
                    n_step_buffer.popleft()
                break
        
        total_rewards.append(current_episode_reward)
        avg_reward = np.mean(total_rewards[-100:])

        if avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            torch.save(agent.policy_net.state_dict(), model_save_path)
            wandb.log({"best_avg_reward": best_avg_reward}, step=agent.steps_done)
            if not args.quiet:
                print(f"*** New best model saved with avg reward {best_avg_reward:.2f} at episode {episode_index+1} ***")

        log_dict = {
            "episode": episode_index + 1,
            "episode_reward": current_episode_reward,
            "avg_reward_100_episodes": avg_reward,
            "epsilon": agent.eps_threshold,
        }
        wandb.log(log_dict, step=agent.steps_done)

        if not args.quiet and (episode_index + 1) % 10 == 0:
            print(f"Epi {episode_index+1} | Reward: {current_episode_reward:.2f} | Avg Reward (100): {avg_reward:.2f}")
        
        if trial:
            trial.report(avg_reward, episode_index)
            if trial.should_prune():
                wandb.run.summary["state"] = "pruned"
                wandb.finish()
                raise optuna.exceptions.TrialPruned()
    
    print(f"\nFinal Average Score (last 100 episodes): {avg_reward:.2f}")
    wandb.run.summary["final_avg_reward"] = avg_reward
    wandb.run.summary["best_avg_reward"] = best_avg_reward
    wandb.run.summary["model_save_path"] = str(model_save_path)
    wandb.finish()
    return avg_reward
