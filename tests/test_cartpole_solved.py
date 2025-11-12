import unittest
import argparse
import numpy as np
import optuna

from src.trainer import run_training
from src.utils import set_seed

class TestCartPoleSolved(unittest.TestCase):
    """
    ### CartPole-v1 해결 능력 검증 테스트
    
    #### 설계 의도 (Design Intent)
    - **통합 테스트(Integration Test):** 에이전트, 모델, 트레이너 등 여러 컴포넌트가 올바르게 통합되어
      실제로 '학습'이라는 목표를 달성할 수 있는지 종합적으로 검증.
    - **성능 기준선 확인:** `CartPole-v1` 환경은 강화학습 에이전트의 성능을 가늠하는 기본적인 벤치마크.
      이 환경을 '해결(solve)'하는지 테스트함으로써, 구현된 알고리즘이 최소한의 성능 기준을 만족하는지 확인.
      (Gymnasium에서 CartPole-v1의 해결 기준은 100 에피소드 연속 평균 보상 195점 이상)
    """

    def test_cartpole_solves(self):
        """
        #### 테스트 목표
        - DQNAgent가 특정 하이퍼파라미터 조합(Double DQN + Dueling DQN)으로 `CartPole-v1` 환경을
          300 에피소드 내에 해결할 수 있는지(평균 보상 195점 이상 달성)를 검증.
        
        #### 테스트 절차
        1. 테스트에 사용할 하이퍼파라미터 `args`를 정의.
        2. 재현성을 위해 시드를 고정.
        3. `run_training` 함수를 호출하여 실제 학습을 진행.
        4. 학습 완료 후 반환된 최종 점수(`final_score`)가 해결 기준인 195점 이상인지 `assertGreaterEqual`로 확인.
        """
        # 1. 테스트를 위한 하이퍼파라미터 설정
        args = argparse.Namespace(
            env_name='CartPole-v1',
            num_episodes=300,
            batch_size=128,
            lr=1e-4,
            gamma=0.99,
            tau=0.005,
            n_steps=1,
            # 성능이 검증된 기본 조합으로 테스트
            use_double=True,
            use_dueling=True,
            use_per=False,
            use_noisy=False,
            use_distributional=False,
            num_atoms=51,
            v_min=-10.0,
            v_max=10.0,
            seed=42,
            quiet=True,
            search=False, # 하이퍼파라미터 탐색 모드 비활성화
            wandb_disable=True, # 테스트 중에는 W&B 로깅 비활성화
        )
        
        # 2. 시드 고정
        set_seed(args.seed)
        
        # 3. 학습 실행
        # run_training 함수는 최종 평균 보상 점수 하나만 반환.
        final_score = run_training(args)
        
        print(f"CartPole 해결 테스트 최종 점수: {final_score}")
        
        # 4. 결과 검증
        self.assertGreaterEqual(final_score, 195.0, "에이전트가 CartPole-v1 환경을 해결하지 못했습니다 (평균 보상 195점 미달).")

if __name__ == '__main__':
    unittest.main()
