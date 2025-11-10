import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import numpy as np
from collections import namedtuple

from .models import QNetwork, DuelingQNetwork, DQN_CNN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

class DQNAgent:
    """
    DQN 에이전트 클래스.
    
    정책망(policy network)과 타겟망(target network)을 포함.
    행동 선택, 경험 저장, 신경망 학습 및 업데이트 전 과정 관리.
    Double DQN, Dueling Networks, PER, N-Step, Noisy Nets, Distributional RL 등
    다양한 Rainbow 구성 요소 지원.
    """
    def __init__(self, state_dim, action_dim, args):
        """
        DQNAgent 초기화.
        
        Args:
            state_dim (int or tuple): 환경 상태 공간 차원.
            action_dim (int): 환경 행동 공간 차원.
            args (argparse.Namespace): 하이퍼파라미터 및 설정값 객체.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 하이퍼파라미터 및 플래그 설정
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.n_steps = args.n_steps
        self.use_double = args.use_double
        self.use_per = args.use_per
        self.use_noisy = args.use_noisy
        self.use_distributional = args.use_distributional
        
        # 환경 타입에 따른 설정
        self.is_atari = "NoFrameskip" in args.env_name
        
        # 신경망 초기화
        # 환경 타입(Atari 여부)과 Dueling 옵션에 따라 적절한 모델 선택.
        if self.is_atari:
            self.policy_net = DQN_CNN(action_dim, use_noisy=self.use_noisy, use_distributional=self.use_distributional, num_atoms=args.num_atoms).to(self.device)
            self.target_net = DQN_CNN(action_dim, use_noisy=self.use_noisy, use_distributional=self.use_distributional, num_atoms=args.num_atoms).to(self.device)
        elif args.use_dueling:
            self.policy_net = DuelingQNetwork(state_dim, action_dim, use_noisy=self.use_noisy, use_distributional=self.use_distributional, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max).to(self.device)
            self.target_net = DuelingQNetwork(state_dim, action_dim, use_noisy=self.use_noisy, use_distributional=self.use_distributional, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max).to(self.device)
        else:
            self.policy_net = QNetwork(state_dim, action_dim, use_noisy=self.use_noisy, use_distributional=self.use_distributional, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max).to(self.device)
            self.target_net = QNetwork(state_dim, action_dim, use_noisy=self.use_noisy, use_distributional=self.use_distributional, num_atoms=args.num_atoms, v_min=args.v_min, v_max=args.v_max).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # 타겟 네트워크는 평가 모드로 설정.

        # 옵티마이저 설정
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)

        # 리플레이 버퍼 초기화
        # PER 사용 여부에 따라 표준 버퍼 또는 우선순위 버퍼 선택.
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(100000)
        else:
            self.memory = ReplayBuffer(100000)

        # Epsilon-greedy를 위한 스텝 카운터
        self.steps_done = 0
        self.eps_threshold = 0

        # Distributional DQN을 위한 설정
        if self.use_distributional:
            self.v_min = args.v_min
            self.v_max = args.v_max
            self.num_atoms = args.num_atoms
            self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

    def _get_greedy_action(self, state):
        """
        현재 정책망 기반 탐욕적인(greedy) 행동 선택.
        
        Args:
            state (torch.Tensor): 현재 상태.
        
        Returns:
            torch.Tensor: Q-value가 가장 높은 행동.
        """
        with torch.no_grad():
            # Atari 이미지는 0-1 사이로 정규화.
            processed_state = state / 255.0 if self.is_atari else state
            
            if self.use_distributional:
                # 분포의 기댓값을 Q-value로 사용하여 행동 선택.
                q_dist = self.policy_net(processed_state).exp()
                q_values = (q_dist * self.support).sum(2)
                return q_values.max(1)[1].view(1, 1)
            else:
                # 표준 DQN에서는 Q-value를 바로 사용.
                return self.policy_net(processed_state).max(1)[1].view(1, 1)

    def select_action(self, state, eps_start, eps_end, eps_decay, evaluation_mode=False):
        """
        주어진 상태에 대해 행동 선택.
        Epsilon-greedy, Noisy Nets, 또는 평가 모드에 따라 다르게 동작.
        
        Args:
            state (torch.Tensor): 현재 상태.
            eps_start, eps_end, eps_decay: Epsilon-greedy 파라미터.
            evaluation_mode (bool): 평가 모드 활성화 여부.
        
        Returns:
            torch.Tensor: 선택된 행동.
        """
        # 평가 모드에서는 항상 탐욕적인 행동 선택.
        if evaluation_mode:
            return self._get_greedy_action(state)

        # Noisy Nets 사용 시, 탐험은 네트워크 자체에서 처리되므로 항상 탐욕적인 행동 선택.
        if self.use_noisy:
            return self._get_greedy_action(state)

        # Epsilon-Greedy 탐험
        sample = random.random()
        self.eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        
        if sample > self.eps_threshold:
            # 탐욕적인 행동 선택
            return self._get_greedy_action(state)
        else:
            # 무작위 행동 선택
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def add_to_memory(self, state, action, reward, next_state, done):
        """
        경험(transition)을 리플레이 버퍼에 추가.
        PER을 사용할 경우, 초기 우선순위를 계산하여 추가.
        """
        if self.use_per:
            # PER의 경우, TD-error를 계산할 수 없으므로 일단 최대 우선순위로 추가.
            # 새로운 경험이 적어도 한 번은 샘플링될 확률 보장.
            # 실제 우선순위는 optimize_model에서 계산되어 업데이트.
            # 이 구현에서는 간단하게 1.0으로 초기화.
            self.memory.add(1.0, state, action, reward, next_state, done)
        else:
            self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        """
        리플레이 버퍼에서 샘플링한 배치로 모델을 한 스텝 학습.
        DQN의 모든 핵심 로직(타겟 계산, 손실 계산, 역전파) 구현.
        """
        if len(self.memory) < self.batch_size:
            return  # 버퍼에 충분한 데이터가 없을 경우 학습 건너뜀.

        # --- 1. 리플레이 버퍼에서 트랜지션 샘플링 ---
        if self.use_per:
            transitions, idxs, is_weights = self.memory.sample(self.batch_size)
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
        
        # 트랜지션 배치를 상태, 행동 등의 텐서로 변환.
        batch = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))(*zip(*transitions))
        
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        # Atari 이미지는 0-1 사이로 정규화.
        if self.is_atari:
            state_batch = state_batch / 255.0
            next_state_batch = next_state_batch / 255.0

        # --- 2. 현재 Q-value 계산: Q(s_t, a_t) ---
        # 정책망을 사용하여 현재 상태-행동 쌍의 Q-value(또는 분포) 계산.
        if self.use_distributional:
            log_q_distribution = self.policy_net(state_batch)
            log_q_s_a = log_q_distribution[range(self.batch_size), action_batch.squeeze(1)]
        else:
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # --- 3. 타겟 Q-value 계산: y_t = r + γ * max_a' Q(s_{t+1}, a') ---
        if self.use_distributional:
            with torch.no_grad():
                # 다음 상태의 Q-value 분포 계산.
                next_log_q_distribution = self.target_net(next_state_batch)
                
                # Double DQN 로직 적용: 정책망으로 최적 행동 선택.
                next_actions = self.policy_net(next_state_batch).exp()
                next_actions = (next_actions * self.support).sum(2).argmax(1)
                
                # 선택된 행동에 해당하는 타겟망의 분포 가져옴.
                next_dist = next_log_q_distribution[range(self.batch_size), next_actions].exp()

                # 타겟 분포 프로젝션 (C51 알고리즘 핵심)
                # Tz = r + γ^n * z
                projected_support = reward_batch.unsqueeze(1) + (self.gamma**self.n_steps) * self.support.unsqueeze(0) * (1 - done_batch.unsqueeze(1))
                projected_support = projected_support.clamp(self.v_min, self.v_max)

                # 프로젝션된 지지대를 원래 지지대의 인덱스로 변환하고, 확률 분배.
                b = (projected_support - self.v_min) / self.delta_z
                l = b.floor().long()
                u = b.ceil().long()

                # 선형 보간을 통해 확률 질량 분배.
                m = torch.zeros_like(next_dist, device=self.device)
                offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(self.device)
                m.view(-1).index_add_(0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1))
                
                target_q_distribution = m
        else:
            # 비-분포(Non-distributional) DQN의 타겟 계산
            with torch.no_grad():
                if self.use_double:
                    # Double DQN: 정책망으로 행동 선택, 타겟망으로 가치 평가.
                    best_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
                    next_state_values = self.target_net(next_state_batch).gather(1, best_actions).squeeze(1)
                else:
                    # 표준 DQN: 타겟망으로 행동 선택 및 가치 평가.
                    next_state_values = self.target_net(next_state_batch).max(1)[0]
            
            # 최종 상태가 아닌 경우에만 다음 상태의 가치 고려.
            next_state_values[done_batch] = 0.0
            # 벨만 방정식에 따라 타겟 Q-value 계산.
            expected_state_action_values = (next_state_values * (self.gamma**self.n_steps)) + reward_batch

        # --- 4. 손실(Loss) 계산 ---
        if self.use_distributional:
            # 분포 DQN: 타겟 분포와 현재 분포 사이의 Cross-Entropy Loss 계산.
            loss = - (target_q_distribution * log_q_s_a).sum(1)
        else:
            # 표준 DQN: Smooth L1 Loss (Huber Loss) 사용.
            criterion = nn.SmoothL1Loss(reduction='none' if self.use_per else 'mean')
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # PER을 사용할 경우, 중요도 샘플링 가중치를 손실에 곱하여 편향 보정.
        if self.use_per:
            # 새로운 TD-error 계산하여 버퍼의 우선순위 업데이트.
            if self.use_distributional:
                # 분포 DQN의 경우, KL-divergence 자체가 error로 사용될 수 있음.
                errors = loss.detach().cpu().numpy()
            else:
                errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
            
            for i in range(self.batch_size):
                self.memory.update(idxs[i], errors[i])
            
            # 가중치를 적용하여 최종 손실 계산.
            loss = (loss.squeeze(1) * is_weights).mean()

        # --- 5. 모델 최적화 ---
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑을 통해 학습 안정성 향상.
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self, tau):
        """
        타겟 네트워크를 소프트 업데이트 방식으로 업데이트.
        θ'_t = τ*θ_t + (1-τ)*θ'_t
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)
