import torch
import gymnasium as gym
from itertools import count
import wandb
import os
from collections import deque
import numpy as np

from .agent import DQNAgent
from .utils import set_seed, create_environment

def run_training(args):
    """
    전체 학습 과정을 관리하고 실행.
    
    환경과 에이전트 초기화, 학습 루프 실행,
    W&B 로깅, 모델 저장 등의 부가 작업 처리.
    """
    # --- 1. 초기 설정: 학습 환경 구성 ---
    
    # W&B(Weights & Biases) 로거 초기화.
    # 고유한 실행 ID 생성, 모든 하이퍼파라미터(`args`) W&B 서버에 기록.
    # 모든 실험 재현 및 비교 분석 가능.
    run_id = wandb.util.generate_id()
    wandb.init(
        project=args.wandb_project,
        config=vars(args),
        id=run_id,
        name=f"{args.env_name}_{run_id}",
        reinit=True,
        mode="disabled" if args.wandb_disable else "online"
    )

    # 환경 생성.
    # `create_environment` 팩토리 함수는 환경 이름에 따라 Atari 래퍼 적용 등
    # 필요한 전처리 내부적으로 처리. 훈련 코드는 특정 환경에 종속되지 않음.
    is_atari = "NoFrameskip" in args.env_name
    env = create_environment(args.env_name, args.seed)
    # 재현성을 위해 모든 랜덤 시드 고정.
    set_seed(args.seed)

    # 상태 및 행동 공간 차원 가져옴.
    state_dim = env.observation_space.shape if is_atari else env.observation_space.shape[0]
    action_dim = env.action_space.n

    # DQNAgent 초기화.
    # 신경망(policy/target), 옵티마이저, 리플레이 버퍼 생성.
    agent = DQNAgent(state_dim, action_dim, args)

    # 학습된 모델 저장 디렉토리 고유한 실행 ID로 생성.
    # 여러 번의 실행 결과가 서로 덮어쓰는 것 방지.
    model_dir = f"results/models/{run_id}"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- 2. 학습 루프: 에이전트와 환경의 상호작용 ---
    total_rewards = []
    best_avg_reward = -np.inf
    
    # N-Step Learning을 위한 임시 버퍼 초기화. (n_steps > 1일 경우에만 사용)
    n_step_buffer = deque(maxlen=args.n_steps) if args.n_steps > 1 else None

    # 환경에 따른 Epsilon 스케줄 조정.
    # Atari와 같이 복잡한 환경은 더 길고 점진적인 탐험 필요.
    eps_start = 1.0 if is_atari else args.eps_start
    eps_end = 0.1 if is_atari else args.eps_end
    eps_decay = 1000000 if is_atari else args.eps_decay

    # 정해진 수의 에피소드만큼 학습 반복. 에피소드는 시작부터 끝까지의 한 번의 시도.
    for episode_index in range(args.num_episodes):
        state, info = env.reset(seed=args.seed + episode_index)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
        
        current_episode_reward = 0
        
        # 에피소드가 끝날 때까지(terminated or truncated) 스텝 반복.
        for time_step in count():
            # 1. 행동 선택 (Action Selection)
            # 에이전트의 현재 정책에 따라 행동 선택.
            action = agent.select_action(state, eps_start, eps_end, eps_decay)
            
            # 2. 환경과 상호작용 (Environment Step)
            # 선택한 행동을 환경에 전달하고, 다음 상태, 보상, 종료 여부 받음.
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # 3. 데이터 변환 및 저장 (Data Processing)
            reward = torch.tensor([reward], device=agent.device)
            next_state = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(agent.device)
            
            if args.n_steps > 1:
                # N-Step Learning (n_steps > 1): 버퍼를 사용하여 n-step 트랜지션 구성.
                n_step_buffer.append((state, action, reward, next_state, torch.tensor([done], device=agent.device)))

                # N-Step 버퍼가 가득 찼을 때, 비로소 하나의 완전한 n-step 트랜지션 완성.
                if len(n_step_buffer) == args.n_steps:
                    # n-step 보상 계산: G_{t:t+n} = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1}
                    n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(args.n_steps)])
                    
                    # 리플레이 버퍼에 저장할 최종 트랜지션 구성.
                    # (s_t, a_t, G_{t:t+n}, s_{t+n}, done_{t+n})
                    start_state, start_action, _, _, _ = n_step_buffer[0]
                    end_next_state, end_done = n_step_buffer[-1][3], n_step_buffer[-1][4]
                    
                    agent.add_to_memory(start_state, start_action, n_step_return, end_next_state, end_done)
            else:
                # 표준 DQN (n_steps = 1): 현재 트랜지션을 바로 리플레이 버퍼에 추가.
                agent.add_to_memory(state, action, reward, next_state, torch.tensor([done], device=agent.device))

            # 다음 스텝을 위해 현재 상태 업데이트.
            state = next_state
            current_episode_reward += reward.item()

            # 4. 모델 최적화 (Learning Step)
            # 리플레이 버퍼에서 샘플링한 배치 데이터로 신경망 업데이트.
            agent.optimize_model()

            # 5. 타겟 네트워크 업데이트 (Stabilization Step)
            # 학습 안정성을 위해 타겟 네트워크를 정책 네트워크 방향으로 조금씩 이동.
            agent.update_target_net(args.tau)

            # 에피소드 종료 조건 확인 및 루프 탈출.
            if done:
                if args.n_steps > 1:
                    # 에피소드 종료 시, N-Step 버퍼에 아직 처리되지 않은 나머지 트랜지션 모두 처리.
                    while len(n_step_buffer) > 0:
                        n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(len(n_step_buffer))])
                        start_state, start_action, _, _, _ = n_step_buffer[0]
                        end_next_state, end_done = n_step_buffer[-1][3], n_step_buffer[-1][4]
                        agent.add_to_memory(start_state, start_action, n_step_return, end_next_state, end_done)
                        n_step_buffer.popleft()

                total_rewards.append(current_episode_reward)
                # 최근 100개 에피소드 평균 보상 계산하여 성능 지표로 삼음. 단일 에피소드 보상보다 안정적.
                avg_reward = np.mean(total_rewards[-100:])

                # W&B에 학습 지표 기록.
                wandb.log({
                    "reward": current_episode_reward,
                    "avg_reward_100_episodes": avg_reward,
                    "epsilon": agent.eps_threshold,
                    "episode": episode_index
                })

                if not args.quiet:
                    print(f"Episode {episode_index}: Reward = {current_episode_reward}, Avg Reward (100) = {avg_reward:.2f}")

                # 최고 평균 보상 기록한 모델 저장.
                # 초반(100 에피소드 이전)에는 성능 불안정하므로 저장하지 않음.
                if avg_reward > best_avg_reward and episode_index > 100:
                    best_avg_reward = avg_reward
                    best_model_path = os.path.join(model_dir, f"best_model.pth")
                    torch.save(agent.policy_net.state_dict(), best_model_path)
                    # W&B 서버에도 모델 파일 아티팩트로 저장.
                    wandb.save(best_model_path)
                    if not args.quiet:
                        print(f"*** New best model saved with avg reward {best_avg_reward:.2f} ***")
                
                break # 다음 에피소드로 넘어감.
    
    # --- 3. 학습 종료 처리 ---
    print("Training complete.")
    # 환경 리소스 정리.
    env.close()
    # W&B 실행 종료 및 모든 데이터 서버에 동기화.
    wandb.finish()
