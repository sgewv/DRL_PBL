import argparse

def get_args():
    """
    프로젝트 전체에서 사용될 커맨드 라인 인자를 파싱하고 중앙에서 관리.
    
    Returns:
        argparse.Namespace: 파싱된 인자들이 담긴 객체.
    """
    parser = argparse.ArgumentParser(description='A Flexible Framework for Deep Reinforcement Learning')

    # --- 기본 실행 인자 ---
    # 학습 환경, 에피소드 수, 재현성 시드 등 기본적인 실행 옵션 정의.
    parser.add_argument('--env_name', type=str, default='CartPole-v1', help='실행할 Gymnasium 환경의 이름')
    parser.add_argument('--num_episodes', type=int, default=2000, help='학습을 진행할 총 에피소드 수. 늘리면 더 오래 학습하지만, 시간도 더 오래 걸림.')
    parser.add_argument('--seed', type=int, default=42, help='재현성을 위한 랜덤 시드. 동일한 시드는 동일한 결과를 보장.')
    parser.add_argument('--quiet', action='store_true', help='학습 중 상세 로그 출력을 억제하여 실행 속도를 약간 높임.')

    # --- 핵심 하이퍼파라미터 ---
    # 에이전트의 학습 성능에 직접적인 영향을 미치는 주요 하이퍼파라미터 정의.
    parser.add_argument('--lr', type=float, default=1e-4, help='옵티마이저의 학습률 (Learning Rate). [↑: 빠른 학습, 불안정성 증가], [↓: 안정적, 학습 속도 저하]')
    parser.add_argument('--gamma', type=float, default=0.99, help='미래 보상에 대한 할인율 (Discount Factor). [↑(1에 가깝게): 미래지향적, 장기 보상 중시], [↓(0에 가깝게): 근시안적, 즉각적인 보상 중시]')
    parser.add_argument('--batch_size', type=int, default=128, help='학습에 사용할 미니배치의 크기. [↑: 안정적인 학습, 메모리 사용량 증가], [↓: 불안정한 학습, 빠른 업데이트]')
    parser.add_argument('--tau', type=float, default=0.005, help='타겟 네트워크 소프트 업데이트 계수. [↑: 빠른 업데이트, 불안정성 증가], [↓: 느린 업데이트, 안정성 증가]')
    parser.add_argument('--n_steps', type=int, default=1, help='N-Step Learning에서 사용할 스텝 수. [↑: 빠른 보상 전파, 분산 증가], [↓: 안정적, 느린 보상 전파]')

    # --- Epsilon-Greedy 탐험 하이퍼파라미터 ---
    # Noisy Net을 사용하지 않을 경우, 탐험-활용 트레이드오프를 제어하는 파라미터 정의.
    parser.add_argument('--eps_start', type=float, default=0.9, help='Epsilon의 시작 값. [↑: 초기 탐험 증가], [↓: 초기 탐험 감소]')
    parser.add_argument('--eps_end', type=float, default=0.05, help='Epsilon의 최종 값. [↑: 학습 후반에도 탐험 지속], [↓: 학습 후반에 탐험 최소화]')
    parser.add_argument('--eps_decay', type=int, default=1000, help='Epsilon의 감소 속도. [↑: 탐험 기간 증가], [↓: 빠른 활용(exploitation) 전환]')

    # --- 에이전트 기능 플래그 (Rainbow 구성 요소) ---
    # Rainbow 논문의 주요 알고리즘들을 활성화/비활성화하는 불리언 플래그 정의.
    parser.add_argument('--use_double', action='store_true', help='Double DQN 활성화. Q-value 과대평가(overestimation) 문제 완화.')
    parser.add_argument('--use_dueling', action='store_true', help='Dueling Network 아키텍처 활성화. 상태 가치와 행동 이점을 분리하여 학습 효율 향상.')
    parser.add_argument('--use_per', action='store_true', help='Prioritized Experience Replay (PER) 활성화. 중요한 경험을 더 자주 학습하여 효율 향상.')
    parser.add_argument('--use_noisy', action='store_true', help='Noisy Nets 탐험 활성화. Epsilon-greedy를 대체하는 발전된 탐험 기법.')
    parser.add_distributional('--use_distributional', action='store_true', help='Distributional DQN (C51) 활성화. Q-value를 단일 값이 아닌 분포로 학습.')
    # Distributional DQN 전용 파라미터
    parser.add_argument('--num_atoms', type=int, default=51, help='Distributional DQN에서 사용할 아톰(atom)의 수. [↑: 분포 표현 정교, 계산량 증가], [↓: 분포 표현 단순, 계산량 감소]')
    parser.add_argument('--v_min', type=float, default=-10.0, help='Distributional DQN의 Q-value 분포 최소값. 환경의 최소 누적 보상보다 작게 설정.')
    parser.add_argument('--v_max', type=float, default=10.0, help='Distributional DQN의 Q-value 분포 최대값. 환경의 최대 누적 보상보다 크게 설정.')

    # --- Optuna 하이퍼파라미터 탐색 인자 ---
    # Optuna를 사용한 자동 하이퍼파라미터 탐색 관련 설정 정의.
    parser.add_argument('--search', action='store_true', help='Optuna 하이퍼파라미터 탐색 실행')
    parser.add_argument('--n_trials', type=int, default=100, help='Optuna 탐색을 시도할 횟수. [↑: 더 나은 파라미터 발견 확률 증가, 시간 증가], [↓: 빠른 탐색, 최적해 놓칠 수 있음]')
    parser.add_argument('--num_episodes_per_trial', type=int, default=300, help='Optuna의 각 trial 당 학습할 에피소드 수. [↑: 신뢰도 높은 평가, 시간 증가], [↓: 빠른 평가, 평가 노이즈 증가]')
    parser.add_argument('--search_mode', type=str, default='base', choices=['base', 'all'], help='탐색할 파라미터 범위 선택. (base: 기본, all: 전체)')

    # --- W&B 로깅 인자 ---
    # Weights & Biases 연동 및 실험 트래킹 관련 설정 정의.
    parser.add_argument('--wandb_project', type=str, default='drl-pbl-lecture', help='W&B 프로젝트 이름')
    parser.add_argument('--wandb_disable', action='store_true', help='W&B 로깅 비활성화')

    # --- 평가 모드 인자 ---
    # 학습된 에이전트의 성능을 정성적으로 관찰하기 위한 평가 모드 관련 설정 정의.
    parser.add_argument('--evaluate', action='store_true', help='평가 모드 실행')
    parser.add_argument('--load_model_path', type=str, default=None, help='평가에 사용할 저장된 모델 파일의 경로')
    parser.add_argument('--eval_episodes', type=int, default=10, help='평가를 진행할 에피소드 수')

    return parser.parse_args()