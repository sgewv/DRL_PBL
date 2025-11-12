import gymnasium as gym
import random
import numpy as np
import torch

def set_seed(seed):
    """
    ### 재현성을 위한 랜덤 시드 고정
    
    #### 설계 의도 (Design Intent)
    - 실험의 재현성(Reproducibility): 강화학습 실험은 무작위성에 크게 의존 (e.g., 가중치 초기화, 환경의 무작위성, 탐험).
      동일한 시드를 사용하면 항상 동일한 결과를 얻을 수 있어, 특정 변경 사항의 효과를 정확히 비교하고 실험 결과를 신뢰할 수 있게 됨.
    - 일관된 환경 제공: `random`, `numpy`, `torch` 등 프로젝트에서 사용되는 모든 주요 라이브러리의 시드를 한 번에 설정하여 일관성을 보장.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # CUDA 연산의 결정론적 동작을 보장. 성능은 약간 저하될 수 있으나 재현성을 위해 필요.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_environment(env_name, seed=None):
    """
    ### Gymnasium 환경 생성 및 래핑(Wrapping)
    
    #### 설계 의도 (Design Intent)
    - 환경 생성 표준화: 환경 생성 로직을 중앙에서 관리하여 코드 중복을 피하고 일관성을 유지.
    - 전처리 추상화: 특정 환경(e.g., Atari)에 필요한 복잡한 전처리 과정을 '래퍼(Wrapper)'를 통해 추상화.
      `trainer.py`와 같은 상위 모듈은 환경의 종류에 신경 쓸 필요 없이 표준화된 인터페이스(`step`, `reset`)만 사용하면 됨.
      
    #### Atari 환경 래퍼 설명
    - `AtariPreprocessing`: Atari 게임 환경에 필요한 표준적인 전처리를 수행.
      - Frame Skipping: 여러 프레임 동안 같은 행동을 반복하여 연산 효율을 높임.
      - Grayscale: 이미지를 흑백으로 변환하여 데이터 차원을 줄임.
      - Screen Resizing: 화면 크기를 84x84로 줄여 신경망의 입력 크기를 줄임.
      - No-op Starts: 에피소드 시작 시 랜덤한 횟수(최대 30)만큼 아무 행동도 하지 않아, 에이전트가 다양한 시작 상태에서 학습하도록 함.
    - `FrameStackObservation`: 여러 개의 연속된 프레임(여기서는 4개)을 하나의 상태로 묶음.
      이를 통해 에이전트는 단일 이미지에서는 알 수 없는 동적인 정보(e.g., 공의 이동 방향)를 파악 가능.
    """
    # 최신 Gymnasium 규칙에 따라, "ALE/" 네임스페이스로 Atari 환경을 탐지.
    is_atari = "ALE/" in env_name
    
    if is_atari:
        # `gym.make`를 호출하기 전에 `ale_py`를 임포트하여 Atari 환경을 Gymnasium에 등록.
        import ale_py
    
    env = gym.make(env_name)
    
    if is_atari:
        env = gym.wrappers.AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
        env = gym.wrappers.FrameStackObservation(env, 4)
    
    if seed is not None:
        # 환경 내부의 무작위성(e.g., reset 시 초기 상태)도 시드를 통해 제어.
        # 참고: Gymnasium 1.0.0a2 버전부터는 env.reset(seed=...)를 권장.
        # 여기서는 하위 호환성을 위해 env.seed()도 함께 사용할 수 있음.
        # 이 코드에서는 reset 시 시드를 제공하므로 이 부분은 중복될 수 있으나, 명시적으로 추가.
        env.action_space.seed(seed)
        
    return env