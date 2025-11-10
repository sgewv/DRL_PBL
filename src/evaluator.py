import torch
import gymnasium as gym
from itertools import count
import numpy as np
import os

from .agent import DQNAgent
from .models import QNetwork, DuelingQNetwork, DQN_CNN
from .utils import set_seed, create_environment

def evaluate_agent(args):
    """
    사전 학습된 에이전트를 평가 환경에서 실행하고 상세 로그 출력.
    
    저장된 모델 로드, 탐험 없이(exploration-free) 에이전트 행동 관찰.
    각 스텝의 결정과 결과 정성적 분석을 위해 상세 기록.
    """
    # --- 1. 초기 설정 ---
    
    # 환경 생성.
    # `create_environment` 팩토리 함수는 환경 이름에 따라 적절한 래퍼(wrapper) 적용.
    is_atari = "NoFrameskip" in args.env_name
    env = create_environment(args.env_name, args.seed)
    # 재현성을 위해 모든 랜덤 시드 고정.
    set_seed(args.seed)

    # 상태 및 행동 공간 차원 가져옴.
    state_dim = env.observation_space.shape if is_atari else env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 에이전트 초기화.
    # 평가 모드이므로, 학습 시 사용했던 동일한 설정(use_dueling, use_noisy 등)으로 초기화 필요.
    agent = DQNAgent(state_dim, action_dim, args)

    # 저장된 모델 로드.
    if args.load_model_path:
        print(f"Loading model from: {args.load_model_path}")
        # 모델의 state_dict를 로드하여 정책망에 적용.
        agent.policy_net.load_state_dict(torch.load(args.load_model_path, map_location=agent.device))
        agent.policy_net.eval() # 평가 모드로 설정 (Dropout, BatchNorm 등이 평가 모드로 동작)
        agent.target_net.load_state_dict(agent.policy_net.state_dict()) # 타겟망도 동일하게 설정
        agent.target_net.eval() # 타겟망도 평가 모드로 설정
    else:
        print("No model path provided for evaluation. Using randomly initialized agent.")
        return

    print("\n--- Starting Evaluation for {} episodes ---".format(args.eval_episodes))
    
    total_rewards = []

    # --- 2. 평가 루프 ---
    # 정해진 수의 에피소드만큼 평가 반복.
    for episode_index in range(args.eval_episodes):
        state, info = env.reset(seed=args.seed + episode_index)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
        
        current_episode_reward = 0
        user_persona = info.get('user_persona', 'N/A') # MovieRec-v1 환경에서만 제공되는 정보
        
        print(f"\n--- Episode {episode_index + 1} | User Persona: {user_persona} ---")

        # 에피소드가 끝날 때까지(terminated or truncated) 스텝 반복.
        for time_step in count():
            # 행동 선택 (평가 모드에서는 탐험 없이 항상 탐욕적인 행동 선택.)
            action = agent.select_action(state, args.eps_start, args.eps_end, args.eps_decay, evaluation_mode=True)
            
            # 환경과 상호작용
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # 보상과 다음 상태를 텐서로 변환.
            reward_tensor = torch.tensor([reward], device=agent.device)
            state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(agent.device)
            current_episode_reward += reward_tensor.item()

            # 상세 로그 출력 (정성적 분석에 유용)
            action_name = info.get('action_name', action.item()) # MovieRec-v1 환경에서만 제공되는 정보
            clicked = info.get('clicked', 'N/A')
            fatigue = info.get('fatigue', 'N/A')
            churned = info.get('churned', False)

            print(f"  Step {time_step + 1}:")
            print(f"    - Action: {action_name}")
            print(f"    - Outcome: Clicked={clicked}, Reward={reward:.2f}")
            print(f"    - User State: Fatigue={fatigue:.2f}")
            
            # 에피소드 종료 조건 확인.
            if done:
                if churned:
                    print(f"    - SESSION END: User churned due to high fatigue!")
                else:
                    print(f"    - SESSION END: Episode finished.")
                break
        
        total_rewards.append(current_episode_reward)
        print(f"--- Episode {episode_index + 1} Total Reward: {current_episode_reward:.2f} ---")
    
    # --- 3. 평가 종료 처리 ---
    print("\n--- Evaluation Complete ---")
    print(f"Average Reward over {args.eval_episodes} episodes: {np.mean(total_rewards):.2f}")
    env.close()
