
# Modular Deep Reinforcement learning Framework
A modular PyTorch implementation of a DQN agent supporting key Rainbow components and Optuna-based hyperparameter search.
Features
  - [x] Standard DQN
  
  - [x] Double DQN
  
  - [x] Dueling Networks
  
  - [x] Prioritized Experience Replay (PER)
  
  - [x] Noisy Nets
  
  - [x] Distributional RL (C51)
  
  - [x] N-Step Learning
  
  - [x] Optuna Hyperparameter Optimization

Quick Start
Clone the repo:

git clone https://github.com/sgewv/DRL_PBL.git

Install core dependencies: (Anaconda environment is recommended)

pip install requirements.txt

# Run DQN on CartPole
python main.py --search --search_mode all

python main.py --search --search_mode base

python main.py --search --search_mode double

python main.py --search --search_mode dueling

python main.py --search --search_mode per

python main.py --search --search_mode noisy

python main.py --search --search_mode distributional

python main.py --search --search_mode rainbow

# Reference
Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. nature, 518(7540), 529-533.

Mnih, V., Kavukcuoglu, K., Silver, D., Graves, A., Antonoglou, I., Wierstra, D., & Riedmiller, M. (2013). Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602.

Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

Schaul, T., Quan, J., Antonoglou, I., & Silver, D. (2015). Prioritized experience replay. arXiv preprint arXiv:1511.05952.

Hessel, M., Modayil, J., Van Hasselt, H., Schaul, T., Ostrovski, G., Dabney, W., ... & Silver, D. (2018, April). Rainbow: Combining improvements in deep reinforcement learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 32, No. 1).


