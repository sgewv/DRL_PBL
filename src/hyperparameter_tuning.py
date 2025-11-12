import optuna
import argparse

from .trainer import run_training
from .utils import set_seed


def objective(trial, args):
    """
    ### Optuna를 위한 목적 함수(Objective Function)

    #### 설계 의도 (Design Intent)
    - 단일 시도(Trial) 정의: Optuna가 최적화할 단일 '시도(trial)'의 전체 과정을 정의.
      하나의 trial은 '특정 하이퍼파라미터 조합으로 모델을 학습하고, 그 성능(점수)을 반환하는' 과정.
    - 하이퍼파라미터 제안: `trial` 객체를 사용하여 Optuna가 제안하는 하이퍼파라미터 값들을 받아옴.
      `suggest_float`, `suggest_categorical` 등은 미리 정의된 범위 내에서 최적의 값을 찾기 위한 탐색을 수행.

    Args:
        trial (optuna.Trial): Optuna에 의해 관리되는 단일 시도 객체.
        args (argparse.Namespace): 기본 하이퍼파라미터 및 설정을 담고 있는 객체.

    Returns:
        float: 해당 trial에서 달성한 최종 점수(e.g., 마지막 100개 에피소드의 평균 보상). Optuna는 이 값을 최대화하는 것을 목표로 함.
    """
    # === 하위 목표 1: Trial별 인자 설정 ===
    # 기본 args를 복사하여 이번 trial만의 새로운 인자(trial_args)를 생성.
    trial_args = argparse.Namespace(**vars(args))

    # Optuna가 제안하는 하이퍼파라미터 값으로 trial_args를 덮어씀.
    # `log=True`는 로그 스케일(logarithmic scale)로 값을 탐색하여, 1e-5 ~ 1e-3 와 같이 넓은 범위의 학습률을 효율적으로 탐색.
    trial_args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial_args.gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    trial_args.tau = trial.suggest_float("tau", 0.001, 0.01, log=True)

    # 'all' 모드에서는 Double DQN, Dueling DQN 사용 여부도 탐색 대상에 포함.
    if args.search_mode == 'all':
        trial_args.use_double = trial.suggest_categorical(
            "use_double", [True, False])
        trial_args.use_dueling = trial.suggest_categorical(
            "use_dueling", [True, False])

    # 탐색 중에는 상세 로그를 끄고, 정해진 에피소드 수만큼만 학습하여 빠른 평가를 진행.
    trial_args.quiet = True
    trial_args.num_episodes = args.num_episodes_per_trial

    # === 하위 목표 2: 학습 실행 및 결과 반환 ===
    # 설정된 하이퍼파라미터로 학습을 실행하고 최종 점수를 반환.
    # `run_training` 함수는 Optuna의 Pruning 콜백을 위해 `trial` 객체를 전달받을 수 있도록 수정됨.
    final_score = run_training(trial_args, trial)
    return final_score


def start_optuna_search(args):
    """
    ### Optuna 하이퍼파라미터 탐색 시작

    #### 설계 의도 (Design Intent)
    - 자동화된 최적화: 개발자가 수동으로 하이퍼파라미터를 조정하는 대신, Optuna 프레임워크를 사용하여
      최적의 하이퍼파라미터 조합을 체계적이고 자동화된 방식으로 탐색.

    #### 핵심 아이디어 (Core Idea)
    - Study: 전체 최적화 과정을 관리하는 객체. 어떤 하이퍼파라미터 조합이 어떤 점수를 기록했는지 모든 trial의 이력을 저장.
      `direction="maximize"`는 `objective` 함수가 반환하는 점수를 '최대화'하는 것을 목표로 설정.
    - Pruner (가지치기): 성능이 좋지 않을 것으로 예상되는 trial을 조기에 중단시켜 탐색 시간을 절약하는 기능.
      `MedianPruner`는 동일 스텝에서 다른 trial들의 중간값보다 낮은 성능을 보이는 trial을 중단시킴.
    - Optimize: `objective` 함수를 `n_trials` 횟수만큼 호출하여 전체 최적화 과정을 실행.
    """
    set_seed(args.seed)

    # 1. Study 객체 생성.
    study = optuna.create_study(
        direction="maximize", pruner=optuna.pruners.MedianPruner())

    # 2. `objective` 함수를 `n_trials` 만큼 실행하여 최적화 시작.
    def _optuna_objective(trial: optuna.Trial) -> float:
      return float(objective(trial, args))

    study.optimize(_optuna_objective, n_trials=args.n_trials)

    # 3. 최적화 완료 후 결과 출력.
    print("\n--- 최적화 완료 ---")
    print(f"'{args.search_mode}' 모드에 대한 최적 Trial:")
    best_trial = study.best_trial
    print(f"  - 점수 (Value): {best_trial.value}")
    print("  - 최적 파라미터 (Params): ")
    for key, value in best_trial.params.items():
        print(f"    - {key}: {value}")
