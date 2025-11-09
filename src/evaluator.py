import gymnasium as gym
import torch
import numpy as np
from itertools import count

# Register our custom environments
import src.custom_envs

from .agent import DQNAgent
from .utils import set_seed, create_environment

def evaluate_agent(args):
    """Runs the evaluation loop."""
    if not args.load_model_path:
        print("Error: --load_model_path must be specified for evaluation.")
        return

    is_atari = "NoFrameskip" in args.env_name
    env = create_environment(args.env_name, args.seed)
    set_seed(args.seed)

    state_dim = env.observation_space.shape if is_atari else env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim, action_dim, is_atari,
        use_dueling=args.use_dueling, use_noisy=args.use_noisy,
        use_distributional=args.use_distributional, num_atoms=args.num_atoms,
        v_min=args.v_min, v_max=args.v_max
    )

    print(f"Loading model from: {args.load_model_path}")
    agent.policy_net.load_state_dict(torch.load(args.load_model_path, map_location=agent.device))
    agent.policy_net.eval()

    total_rewards = []
    print(f"\n--- Starting Evaluation for {args.eval_episodes} episodes ---")

    for episode_index in range(args.eval_episodes):
        state, info = env.reset(seed=args.seed + episode_index)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
        
        current_episode_reward = 0
        print(f"\n--- Episode {episode_index + 1} | User Persona: {info.get('user_persona', 'N/A')} ---")
        
        for time_step in count():
            # Select action greedily
            action = agent.select_action(state, args.eps_start, args.eps_end, args.eps_decay, evaluation_mode=True)
            observation, reward, terminated, truncated, info = env.step(action.item())
            
            current_episode_reward += reward
            done = terminated or truncated

            print(f"  Step {time_step+1}:")
            print(f"    - Action: Recommend '{info.get('recommended_movie', 'N/A')}'")
            print(f"    - Outcome: Clicked={info.get('is_clicked', 'N/A')}, Reward={reward:.2f}")
            print(f"    - User State: Fatigue={info.get('fatigue', 0.0):.2f}")
            if done:
                if info.get('churned', False):
                    print("    - SESSION END: User churned due to high fatigue!")
                elif truncated:
                    print("    - SESSION END: Max session length reached.")
                else:
                    print("    - SESSION END: Episode finished.")

            if done:
                break
            else:
                state = torch.from_numpy(np.array(observation)).float().unsqueeze(0).to(agent.device)
        
        total_rewards.append(current_episode_reward)
        print(f"--- Episode {episode_index + 1} Total Reward: {current_episode_reward:.2f} ---")

    avg_reward = np.mean(total_rewards)
    print(f"\n--- Evaluation Complete ---")
    print(f"Average Reward over {args.eval_episodes} episodes: {avg_reward:.2f}")
