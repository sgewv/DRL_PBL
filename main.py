import src.custom_envs
from src.config import get_args
from src.trainer import run_training
from src.evaluator import evaluate_agent
from src.hyperparameter_tuning import start_optuna_search
from src.utils import set_seed

def main():
    """
    ### 설계 의도 (Design Intent)
    
    이 스크립트의 메인 진입점(Entry Point) 역할.
    사용자로부터 커맨드 라인 인자(command-line arguments)를 받아, 
    프로그램의 실행 모드를 결정하는 기능을 수행.
    
    - **중앙 제어:** `config.py`에서 모든 설정을 가져오고, `utils.py`에서 시드를 설정하는 등
      전체 프로그램의 초기 설정을 중앙에서 관리.
    - **모드 분기:** 사용자가 전달한 인자(`--evaluate`, `--search` 등)에 따라 
      '학습', '평가', '하이퍼파라미터 탐색' 중 적절한 모듈을 호출.
    """
    # 1. `config.py`로부터 모든 커맨드 라인 인자와 하이퍼파라미터를 파싱.
    args = get_args()
    
    # 2. 재현성을 위해 랜덤 시드를 고정.
    set_seed(args.seed)

    # 3. 사용자 인자에 따라 적절한 실행 모드를 호출.
    if args.evaluate:
        # 평가 모드: 학습된 에이전트의 성능을 평가.
        evaluate_agent(args)
    elif args.search:
        # 하이퍼파라미터 탐색 모드: Optuna를 사용해 최적의 하이퍼파라미터 조합을 탐색.
        start_optuna_search(args)
    else:
        # 기본 모드: 에이전트 학습을 실행.
        run_training(args)

if __name__ == '__main__':
    # `python main.py` 실행 시 이 부분이 호출됨.
    main()
