import optuna
import argparse

from .trainer import run_training
from .utils import set_seed

def objective(trial, args):
    """Defines a single trial for Optuna to optimize."""
    trial_args = argparse.Namespace(**vars(args))
    trial_args.lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    trial_args.gamma = trial.suggest_float("gamma", 0.9, 0.999, log=True)
    trial_args.tau = trial.suggest_float("tau", 0.001, 0.01, log=True)
    
    if args.search_mode == 'all':
        trial_args.use_double = trial.suggest_categorical("use_double", [True, False])
        trial_args.use_dueling = trial.suggest_categorical("use_dueling", [True, False])

    trial_args.quiet = True
    trial_args.num_episodes = args.num_episodes_per_trial
    final_score = run_training(trial_args, trial)
    return final_score

def start_optuna_search(args):
    """Starts the Optuna hyperparameter search."""
    set_seed(args.seed)
    study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
    objective_func = lambda trial: objective(trial, args)
    study.optimize(objective_func, n_trials=args.n_trials)

    print("\n--- Optimization Finished ---")
    print(f"Best trial for mode '{args.search_mode}':")
    trial = study.best_trial
    print(f"  Value (Score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
