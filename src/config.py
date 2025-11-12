import argparse


def get_args():
    """
    ### 설계 의도 (Design Intent)

    프로젝트 전체에서 사용될 모든 하이퍼파라미터와 실행 옵션을 중앙에서 관리.
    `argparse` 라이브러리를 사용하여 커맨드 라인에서 직접 설정을 변경할 수 있도록 유연성을 제공.

    - 중앙화(Centralization): 모든 설정이 한 곳에 모여 있어, 파라미터의 조회, 수정, 관리가 용이.
      코드 여러 곳에 하드코딩된 값을 퍼뜨리는 것을 방지하여 일관성을 유지.
    - 유연성(Flexibility): 코드를 직접 수정하지 않고도 커맨드 라인 인자를 통해
      환경, 알고리즘, 하이퍼파라미터 등을 쉽게 변경하며 다양한 실험을 수행 가능.
    """
    parser = argparse.ArgumentParser(
        description='A Flexible Framework for Deep Reinforcement Learning')

    # === 기본 실행 인자 ===
    # 학습 환경, 에피소드 수, 재현성 시드 등 기본적인 실행 옵션.
    parser.add_argument('--env_name', type=str,
                        default='CartPole-v1', help='실행할 Gymnasium 환경의 이름.')
    parser.add_argument('--num_episodes', type=int, default=2000,
                        help='학습을 진행할 총 에피소드 수. 에이전트가 환경과 상호작용하며 학습하는 전체 횟수.')
    parser.add_argument('--seed', type=int, default=42,
                        help='재현성을 위한 랜덤 시드. 동일한 시드는 동일한 결과를 보장하여 실험의 신뢰도를 높임.')
    parser.add_argument('--quiet', action='store_true',
                        help='학습 중 상세 로그 출력을 억제하여 실행 속도를 약간 높임.')

    # === 핵심 하이퍼파라미터 ===
    # 에이전트의 학습 성능과 안정성에 직접적인 영향을 미치는 주요 하이퍼파라미터.
    parser.add_argument('--lr', type=float, default=1e-4,
                        help="학습률(Learning Rate). 신경망 가중치를 업데이트하는 보폭. [↑: 빠른 학습, 불안정성 증가], [↓: 안정적, 학습 속도 저하]")
    parser.add_argument('--gamma', type=float, default=0.99,
                        help="할인율(Discount Factor). 미래 보상을 현재 가치로 환산하는 비율. [↑(1에 가깝게): 미래지향적, 장기 보상 중시], [↓(0에 가깝게): 근시안적, 즉각적인 보상 중시]")
    parser.add_argument('--batch_size', type=int, default=128,
                        help="미니배치 크기. 한 번의 학습 스텝에 사용할 경험 데이터의 수. [↑: 안정적인 그래디언트, 메모리 사용량 증가], [↓: 불안정한 학습, 빠른 업데이트]")
    parser.add_argument('--tau', type=float, default=0.005,
                        help="타겟 네트워크 소프트 업데이트 계수. 정책망의 가중치를 타겟망에 얼마나 부드럽게 반영할지 결정. [↑: 빠른 업데이트, 불안정성 증가], [↓: 느린 업데이트, 안정성 증가]")
    parser.add_argument('--n_steps', type=int, default=1,
                        help="N-Step Learning에서 사용할 스텝 수. 한 번에 몇 스텝 앞의 미래까지 고려하여 타겟을 계산할지 결정. [↑: 빠른 보상 전파, 분산 증가], [↓: 안정적, 느린 보상 전파]")

    # === Epsilon-Greedy 탐험 하이퍼파라미터 ===
    # Noisy Net을 사용하지 않을 경우, '탐험-활용 트레이드오프'를 제어하는 파라미터.
    parser.add_argument('--eps_start', type=float, default=0.9,
                        help="Epsilon의 시작 값. 학습 초기에 무작위 행동을 할 확률. [↑: 초기 탐험 증가], [↓: 초기 탐험 감소]")
    parser.add_argument('--eps_end', type=float, default=0.05,
                        help="Epsilon의 최종 값. 학습이 끝날 무렵 무작위 행동을 할 최소 확률. [↑: 학습 후반에도 탐험 지속], [↓: 학습 후반에 탐험 최소화]")
    parser.add_argument('--eps_decay', type=int, default=1000,
                        help="Epsilon의 감소 속도. Epsilon이 시작 값에서 최종 값까지 얼마나 빠르게 감소할지 결정. [↑: 탐험 기간 증가], [↓: 빠른 활용(exploitation) 전환]")

    # === 에이전트 기능 플래그 (Rainbow DQN 구성 요소) ===
    # Rainbow 논문의 주요 알고리즘들을 활성화/비활성화하는 불리언 플래그.
    parser.add_argument('--use_double', action='store_true',
                        help="Double DQN 활성화. Q-value 과대평가(overestimation) 문제를 완화하여 학습 안정성 향상.")
    parser.add_argument('--use_dueling', action='store_true',
                        help="Dueling Network 아키텍처 활성화. 상태 가치와 행동 이점을 분리하여 학습 효율 향상.")
    parser.add_argument('--use_per', action='store_true',
                        help="Prioritized Experience Replay (PER) 활성화. TD-error가 큰 중요한 경험을 더 자주 학습하여 효율 향상.")
    parser.add_argument('--use_noisy', action='store_true',
                        help="Noisy Nets 탐험 활성화. Epsilon-greedy를 대체하는, 학습을 통해 탐험을 조절하는 발전된 기법.")
    parser.add_argument('--use_distributional', action='store_true',
                        help="Distributional DQN (C51) 활성화. Q-value를 단일 값이 아닌 확률 분포로 학습하여 더 풍부한 정보 학습.")

    # --- Distributional DQN 전용 파라미터 ---
    parser.add_argument('--num_atoms', type=int, default=51,
                        help="Distributional DQN에서 사용할 아톰(atom)의 수. Q-value 분포를 표현하는 이산적인 지지점의 개수. [↑: 분포 표현 정교, 계산량 증가], [↓: 분포 표현 단순, 계산량 감소]")
    parser.add_argument('--v_min', type=float, default=-10.0,
                        help="Distributional DQN의 Q-value 분포 최소값. 환경의 누적 보상 최소값보다 작게 설정해야 함.")
    parser.add_argument('--v_max', type=float, default=10.0,
                        help="Distributional DQN의 Q-value 분포 최대값. 환경의 누적 보상 최대값보다 크게 설정해야 함.")

    # === Optuna 하이퍼파라미터 탐색 인자 ===
    # Optuna를 사용한 자동 하이퍼파라미터 탐색 관련 설정.
    parser.add_argument('--search', action='store_true',
                        help='Optuna 하이퍼파라미터 탐색 실행.')
    parser.add_argument('--n_trials', type=int, default=100,
                        help="Optuna 탐색을 시도할 횟수. [↑: 더 나은 파라미터 발견 확률 증가, 시간 증가], [↓: 빠른 탐색, 최적해 놓칠 수 있음]")
    parser.add_argument('--num_episodes_per_trial', type=int, default=300,
                        help="Optuna의 각 trial 당 학습할 에피소드 수. [↑: 신뢰도 높은 평가, 시간 증가], [↓: 빠른 평가, 평가 노이즈 증가]")
    parser.add_argument('--search_mode', type=str, default='base', choices=[
                        'base', 'all'], help="탐색할 파라미터 범위 선택. (base: lr, gamma, tau / all: base + double, dueling)")

    # === W&B 로깅 인자 ===
    # Weights & Biases 연동 및 실험 트래킹 관련 설정.
    parser.add_argument('--wandb_project', type=str,
                        default='drl-pbl-lecture', help='W&B 프로젝트 이름.')
    parser.add_argument('--wandb_disable',
                        action='store_true', help='W&B 로깅 비활성화.')

    # === 평가 모드 인자 ===
    # 학습된 에이전트의 성능을 관찰하기 위한 평가 모드 관련 설정.
    parser.add_argument('--evaluate', action='store_true', help='평가 모드 실행.')
    parser.add_argument('--load_model_path', type=str,
                        default=None, help='평가에 사용할 저장된 모델(.pth) 파일의 경로.')
    parser.add_argument('--eval_episodes', type=int,
                        default=10, help='평가를 진행할 에피소드 수.')

    return parser.parse_args()
