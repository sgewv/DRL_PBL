from src.config import get_args
from src.trainer import run_training
from src.evaluator import evaluate_agent
from src.hyperparameter_tuning import start_optuna_search
from src.utils import set_seed

def main():
    """Parses arguments and runs the appropriate mode."""
    args = get_args()
    set_seed(args.seed)

    if args.evaluate:
        evaluate_agent(args)
    elif args.search:
        start_optuna_search(args)
    else:
        run_training(args)

if __name__ == '__main__':
    main()
