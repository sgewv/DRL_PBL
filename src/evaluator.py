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
    ### 설계 의도 (Design Intent)
    
    이 함수의 목적은 학습된 에이전트의 '순수한 성능'을 측정하는 것.
    학습 과정에 포함된 무작위 탐험(exploration) 요소를 완전히 배제하고, 
    에이전트가 학습한 정책(policy)에 따라 얼마나 좋은 결정을 내리는지 정량적/정성적으로 평가.
    
    - 정량적 평가: 여러 에피소드에 걸쳐 얻은 평균 보상 점수를 통해 모델의 전반적인 성능을 객관적인 수치로 확인.
    - 정성적 평가: 각 스텝마다 에이전트가 어떤 행동을 선택하고 그 결과가 어떠했는지 상세 로그를 출력.
      이를 통해 에이전트의 행동 패턴을 분석하고, 특정 상황에서 왜 그런 결정을 내리는지 이해하는 데 도움.
    """
    # === 하위 목표 1: 평가 환경 및 에이전트 준비 (Setup Environment & Agent) ===
    
    # 환경 생성 및 시드 고정
    is_atari = "NoFrameskip" in args.env_name
    env = create_environment(args.env_name, args.seed)
    set_seed(args.seed)

    state_dim = env.observation_space.shape if is_atari else env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 에이전트 초기화. 학습 시 사용했던 것과 동일한 아키텍처로 구성.
    agent = DQNAgent(state_dim, action_dim, args)

    # --- 저장된 모델 가중치 로드 ---
    if args.load_model_path:
        print(f"Loading model from: {args.load_model_path}")
        agent.policy_net.load_state_dict(torch.load(args.load_model_path, map_location=agent.device))
        
        # --- 평가 모드 설정 ---
        # 왜 .eval()을 호출하는가?
        #   - Dropout, BatchNorm과 같은 레이어들의 동작을 '평가 모드'로 전환.
        #   - NoisyNet을 사용하는 경우, 노이즈를 비활성화하여 결정론적(deterministic) 행동을 보장.
        #   - 결과적으로, 동일한 상태에 대해 항상 동일한 행동을 선택하게 하여 일관된 성능 측정 가능.
        agent.policy_net.eval()
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        agent.target_net.eval()
    else:
        print("평가할 모델 경로가 지정되지 않음")
        return

    print(f"\n--- {args.eval_episodes} 에피소드에 대한 평가 시작 ---")
    
    total_rewards = []

    # === 하위 목표 2: 평가 루프 실행 (Run Evaluation Loop) ===
    # 정해진 수의 에피소드만큼 평가를 반복하여 평균 성능 측정.
    for episode_index in range(args.eval_episodes):
        state, info = env.reset(seed=args.seed + episode_index)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
        
        current_episode_reward = 0
        user_persona = info.get('user_persona', 'N/A') # MovieRec-v1 환경에서 제공
        
        print(f"\n--- 에피소드 {episode_index + 1} | 사용자 페르소나: {user_persona} ---")

        # 에피소드가 끝날 때까지(done) 스텝 반복.
        for time_step in count():
            # --- 행동 선택 ---
            # `evaluation_mode=True`로 설정하여 탐험(무작위 행동) 없이 항상 최적의 행동(greedy action)만 선택.
            action = agent.select_action(state, args.eps_start, args.eps_end, args.eps_decay, evaluation_mode=True)
            
            # --- 환경과 상호작용 ---
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            
            # --- 다음 상태 준비 ---
            reward_tensor = torch.tensor([reward], device=agent.device)
            state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(agent.device)
            current_episode_reward += reward_tensor.item()

            # --- 정성적 분석을 위한 상세 로그 출력 ---
            action_name = info.get('recommended_movie', action.item())
            clicked = info.get('is_clicked', 'N/A')
            fatigue = info.get('fatigue', 'N/A')
            churned = info.get('churned', False)

            # Format clicked and fatigue conditionally
            clicked_display = "클릭됨" if clicked is True else ("클릭 안됨" if clicked is False else str(clicked))
            fatigue_display = f"{fatigue:.2f}" if isinstance(fatigue, (int, float)) else str(fatigue)

            print(f"  스텝 {time_step + 1}:")
            print(f"    - 선택한 행동: {action_name}")
            print(f"    - 결과: 클릭={clicked_display}, 보상={reward:.2f}")
            print(f"    - 사용자 상태: 피로도={fatigue_display}")
            
            # --- 에피소드 종료 조건 확인 ---
            if done:
                if churned:
                    print(f"    - 세션 종료: 사용자 이탈")
                else:
                    print(f"    - 세션 종료: 최대 스텝에 도달")
                break
        
        total_rewards.append(current_episode_reward)
        print(f"--- 에피소드 {episode_index + 1} 총 보상: {current_episode_reward:.2f} ---")
    
    # === 하위 목표 3: 평가 결과 요약 (Summarize Results) ===
    print("\n--- 평가 완료 ---")
    print(f"{args.eval_episodes} 에피소드 평균 보상: {np.mean(total_rewards):.2f}")
    env.close()
