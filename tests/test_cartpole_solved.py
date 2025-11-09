import unittest
import argparse
import numpy as np

from main import run_training
from src.utils import set_seed

class TestCartPoleSolved(unittest.TestCase):

    def test_cartpole_solves(self):
        """
        Tests if the DQNAgent can achieve an average reward of 195 over 100 episodes
        on CartPole-v1 within 300 episodes.
        """
        args = argparse.Namespace(
            env_name='CartPole-v1',
            num_episodes=300,
            batch_size=128,
            lr=1e-4,
            gamma=0.99,
            tau=0.005,
            n_steps=1,
            early_stop_threshold=195,
            is_atari=False,
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
            search=False,
            search_mode='single_run'
        )
        
        set_seed(args.seed)
        
        # Suppress W&B logs for this test
        import os
        os.environ["WANDB_MODE"] = "disabled"
        
        episode_durations, episode_rewards, final_score = run_training(args, results_dir=None)
        
        print(f"Test finished with final score: {final_score}")
        
        self.assertGreaterEqual(final_score, 195.0, "Agent failed to solve CartPole-v1")

if __name__ == '__main__':
    unittest.main()
