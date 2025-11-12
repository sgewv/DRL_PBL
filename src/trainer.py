import torch
import gymnasium as gym
from itertools import count
import wandb
import os
from collections import deque
import numpy as np

from .agent import DQNAgent
from .utils import set_seed, create_environment

def run_training(args, trial=None):
    """
    ### 설계 의도 (Design Intent)
    
    강화학습의 전체 '학습 루프(training loop)'를 총괄하는 지휘자(conductor) 역할.
    에이전트와 환경 간의 상호작용을 조율하고, 학습 과정을 모니터링하며, 결과를 기록하는 모든 책임을 짐.
    
    - 프로세스 관리: 환경 초기화, 에이전트 생성, 학습 루프 실행, 결과 로깅 및 모델 저장 등
      학습의 시작부터 끝까지 모든 단계를 순차적으로 관리.
    - 추상화: `agent.py`의 `DQNAgent`가 '무엇을' 학습할지(알고리즘)를 결정한다면, `trainer.py`는
      '어떻게' 학습시킬지(프로세스)를 결정. 이를 통해 알고리즘과 학습 절차를 분리.
    """
    # === 하위 목표 1: 학습 환경 구성 (Setup Training Environment) ===
    
    # --- W&B(Weights & Biases) 로거 초기화 ---
    # 왜 W&B를 사용하는가?
    #   - 실험 관리: 모든 하이퍼파라미터, 학습 지표(보상, 손실 등), 시스템 리소스 사용량을 자동으로 기록.
    #   - 재현성 및 비교: 과거 실험을 쉽게 재현하고, 여러 실험 결과를 대시보드에서 시각적으로 비교 분석 가능.
    if not args.search: # 하이퍼파라미터 탐색 중에는 개별 run을 생성하지 않음
        run_id = wandb.util.generate_id()
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            id=run_id,
            name=f"{args.env_name}_{run_id}",
            reinit=True,
            mode="disabled" if args.wandb_disable else "online"
        )

    # --- 환경 및 에이전트 초기화 ---
    is_atari = "ALE/" in args.env_name
    env = create_environment(args.env_name, args.seed)
    set_seed(args.seed) # 재현성을 위해 모든 랜덤 시드 고정

    state_dim = env.observation_space.shape if is_atari else env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim, args)

    # --- 결과 저장 디렉토리 생성 ---
    # 고유한 실행 ID(`run_id`)로 디렉토리를 만들어 여러 실험 결과가 덮어쓰이는 것을 방지.
    if not args.search:
        model_dir = f"results/models/{run_id}"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    # === 하위 목표 2: 학습 루프 실행 (Run Training Loop) ===
    total_rewards = []
    best_avg_reward = -np.inf
    final_avg_reward = 0
    
    # N-Step Learning을 위한 임시 버퍼. `n_steps > 1`일 경우에만 사용.
    n_step_buffer = deque(maxlen=args.n_steps) if args.n_steps > 1 else None

    # 환경 복잡도에 따라 Epsilon 스케줄을 다르게 설정. (Atari는 더 긴 탐험 필요)
    eps_start = 1.0 if is_atari else args.eps_start
    eps_end = 0.1 if is_atari else args.eps_end
    eps_decay = 1000000 if is_atari else args.eps_decay

    # --- 에피소드 반복 ---
    # 에피소드(Episode): 에이전트가 환경의 시작 상태에서 종료 상태에 도달하기까지의 한 번의 시도.
    for episode_index in range(args.num_episodes):
        state, info = env.reset(seed=args.seed + episode_index)
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(agent.device)
        
        current_episode_reward = 0
        
        # --- 스텝 반복 (에피소드 내 상호작용) ---
        for time_step in count():
            # --- 1. 행동 선택 (Action Selection) ---
            action = agent.select_action(state, eps_start, eps_end, eps_decay)
            
            # --- 2. 환경과 상호작용 (Environment Step) ---
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # --- 3. 데이터 변환 및 저장 (Data Processing & Storing) ---
            reward_tensor = torch.tensor([reward], device=agent.device)
            next_state_tensor = torch.from_numpy(np.array(next_state)).float().unsqueeze(0).to(agent.device)
            done_tensor = torch.tensor([done], device=agent.device)
            
            # --- N-Step Learning 로직 ---
            # 왜 N-Step Learning을 사용하는가?
            #   - 1-step TD는 보상을 한 번에 한 스텝씩만 전파하여 학습이 느릴 수 있음.
            #   - N-Step은 N 스텝 후의 미래 보상까지 한 번에 반영하여 학습 속도를 높이고, 더 안정적인 타겟을 제공 가능.
            if args.n_steps > 1:
                n_step_buffer.append((state, action, reward_tensor, next_state_tensor, done_tensor))
                if len(n_step_buffer) == args.n_steps:
                    # N-step 보상(Return) 계산
                    # (수식: G_{t:t+n} = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1})
                    n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(args.n_steps)])
                    
                    # 리플레이 버퍼에 저장할 최종 트랜지션: (s_t, a_t, G_{t:t+n}, s_{t+n}, done_{t+n})
                    start_state, start_action, _, _, _ = n_step_buffer[0]
                    end_next_state, end_done = n_step_buffer[-1][3], n_step_buffer[-1][4]
                    
                    agent.add_to_memory(start_state, start_action, n_step_return, end_next_state, end_done)
            else:
                # 표준 1-Step DQN: 현재 트랜지션을 바로 리플레이 버퍼에 추가.
                agent.add_to_memory(state, action, reward_tensor, next_state_tensor, done_tensor)

            state = next_state_tensor
            current_episode_reward += reward

            # --- 4. 모델 최적화 (Learning Step) ---
            # `agent.py`의 `optimize_model()` 호출. 리플레이 버퍼에서 샘플링한 배치로 신경망 업데이트.
            agent.optimize_model()

            # --- 5. 타겟 네트워크 업데이트 (Stabilization Step) ---
            # `agent.py`의 `update_target_net()` 호출. 학습 안정성을 위해 타겟 네트워크를 부드럽게 업데이트.
            agent.update_target_net(args.tau)

            # --- 에피소드 종료 처리 ---
            if done:
                # N-Step 버퍼에 남아있는 나머지 트랜지션들을 처리.
                if args.n_steps > 1:
                    while len(n_step_buffer) > 0:
                        n_step_return = sum([n_step_buffer[i][2] * (args.gamma**i) for i in range(len(n_step_buffer))])
                        start_state, start_action, _, _, _ = n_step_buffer.popleft()
                        end_next_state, end_done = n_step_buffer[-1][3], n_step_buffer[-1][4] if len(n_step_buffer) > 0 else (next_state_tensor, done_tensor)
                        agent.add_to_memory(start_state, start_action, n_step_return, end_next_state, end_done)

                total_rewards.append(current_episode_reward)
                # 최근 100개 에피소드의 평균 보상을 사용. 단일 에피소드 보상보다 훨씬 안정적인 성능 지표.
                avg_reward = np.mean(total_rewards[-100:])
                final_avg_reward = avg_reward

                # W&B에 학습 지표 기록
                if not args.search:
                    wandb.log({
                        "reward": current_episode_reward,
                        "avg_reward_100_episodes": avg_reward,
                        "epsilon": agent.eps_threshold,
                        "episode": episode_index
                    })

                if not args.quiet:
                    print(f"에피소드 {episode_index}: 보상 = {current_episode_reward}, 평균 보상(100) = {avg_reward:.2f}")

                # --- 최고 성능 모델 저장 ---
                # 왜 특정 조건에서만 저장하는가?
                #   - 학습 초반(e.g., 100 에피소드 이전)에는 성능이 불안정하므로, 우연히 얻은 높은 점수에 속지 않기 위함.
                #   - 안정적인 평균 보상이 이전 최고 기록을 경신했을 때만 저장하여 가장 견고한 모델을 확보.
                if not args.search and avg_reward > best_avg_reward and episode_index > 100:
                    best_avg_reward = avg_reward
                    best_model_path = os.path.join(model_dir, f"best_model.pth")
                    torch.save(agent.policy_net.state_dict(), best_model_path)
                    wandb.save(best_model_path) # W&B 서버에도 모델 파일 아티팩트로 저장
                    if not args.quiet:
                        print(f"*** 새로운 최고 모델 저장됨 (평균 보상: {best_avg_reward:.2f}) ***")
                
                # --- Optuna 가지치기(Pruning) ---
                if trial:
                    trial.report(avg_reward, episode_index)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

                break # 다음 에피소드로.
    
    # === 하위 목표 3: 학습 종료 처리 (Finalize Training) ===
    if not args.search:
        print("학습 완료.")
        env.close()
        wandb.finish()
    
    return final_avg_reward
