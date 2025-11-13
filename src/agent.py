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
    ### 설계 의도 (Design Intent)
    
    에이전트의 핵심 책임인 행동 결정(action selection)을 하는 함수.
    경험 저장(experience storing), 그리고 학습(learning)을 모두 관리.
    
    - 모듈성(Modularity): 다양한 DQN 개선 기법(Double, Dueling, PER 등)을 플래그 형태로 
      쉽게 활성화/비활성화하도록 설계. 특정 기법의 효과를 독립적으로 실험하고 분석하기 용이.
      
    - 추상화(Abstraction): `trainer.py`와 같은 상위 모듈과 상호작용하며, `models.py`나 
      `replay_buffer.py`의 구체적인 구현으로부터 독립적으로 동작. 각 컴포넌트의 유연한 교체 가능.
    """
    def __init__(self, state_dim, action_dim, args):
        """
        ### 설계 의도 (Design Intent)
        
        에이전트의 모든 구성 요소를 초기화하고 준비하는 단계. 로봇 조립처럼 필요한 부품들을 가져와 연결.
        
        - 정책망과 타겟망:
          - 정책망(Policy Network): 현재 상태에서 어떤 행동이 최선일지 결정하는 신경망. 에이전트의 '행동 주체'.
          - 타겟망(Target Network): 학습 과정에서 안정적인 목표(target) Q-value를 제공하는 별도의 신경망. '안정적인 기준점' 역할.
          이 두 네트워크의 분리가 DQN 안정성의 핵심.
        
        - 유연한 아키텍처 선택: `args` 값에 따라 환경 특성(벡터, 이미지)에 맞는 최적의 모델을 동적으로 선택.
          - **Dueling Network:** 상태의 가치(V(s))와 각 행동의 이점(A(s,a))을 분리하여 학습, 더 효율적인 학습을 유도하는 구조.
          
        - 기억 장치 (리플레이 버퍼):
          - 리플레이 버퍼(Replay Buffer): 에이전트의 경험(state, action, reward 등)을 저장하는 메모리. 
            경험 재사용으로 데이터 효율성을 높이고, 샘플 간 상관관계를 깨뜨려 학습 안정화.
          - PER(Prioritized Experience Replay): TD-error가 큰, 즉 '놀라움'이 큰 중요한 경험을 더 자주 샘플링하여 학습 효율을 극대화하는 기법.
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # --- 하이퍼파라미터 및 알고리즘 플래그 설정 ---
        # 에이전트의 학습 전략과 행동 방식을 결정하는 값들.
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.n_steps = args.n_steps
        self.use_double = args.use_double
        self.use_per = args.use_per
        self.use_noisy = args.use_noisy
        self.use_distributional = args.use_distributional
        
        self.is_atari = "ALE/" in args.env_name
        
        # --- 신경망 초기화: '두 개의 뇌' 생성 ---
        # 왜 policy_net과 target_net을 분리하는가?
        #   - 하나의 네트워크만 사용 시, Q-value 업데이트마다 타겟 값(y)이 계속 흔들리는 문제 발생.
        #   - target_net은 한동안 고정되어 '안정적인 학습 타겟'을 제공함으로써 이 문제 해결.
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
        self.target_net.eval()  # target_net은 학습되지 않으며, policy_net으로부터 가중치를 복사받음.

        # --- 옵티마이저 설정 ---
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)

        # --- 리플레이 버퍼 초기화: '기억 저장소' 생성 ---
        # 왜 PER을 사용하는가?
        #   - 모든 경험의 가치는 동일하지 않음. TD-error가 큰 '놀라운' 경험이 학습에 더 중요.
        #   - PER은 이런 중요한 경험을 더 자주 샘플링하여 학습 효율을 극대화.
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(100000)
        else:
            self.memory = ReplayBuffer(100000)

        self.steps_done = 0
        self.eps_threshold = 0

        # --- Distributional DQN을 위한 설정 ---
        # 왜 Q-value를 분포로 학습하는가? (Distributional Reinforcement Learning)
        #   - 미래 보상은 불확실성을 내포한 '확률 변수'. 이를 단일 기댓값(Q-value)으로 예측 시 정보 손실 발생.
        #   - C51 알고리즘은 Q-value를 이산 확률 분포로 모델링하여 더 풍부하고 안정적인 학습을 가능케 함.
        if self.use_distributional:
            self.v_min = args.v_min
            self.v_max = args.v_max
            self.num_atoms = args.num_atoms
            self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(self.device)
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)

    def _get_greedy_action(self, state):
        """
        ### 설계 의도 (Design Intent)
        
        현재 `policy_net`이 학습한 정책에 따라 가장 가치가 높다고 판단되는 행동(greedy action)을 선택.
        `select_action`의 '활용(exploitation)' 부분과 평가(evaluation) 시에 공통으로 사용되어 코드 중복 방지.
        """
        with torch.no_grad():
            processed_state = state / 255.0 if self.is_atari else state
            
            if self.use_distributional:
                # 분포의 기댓값을 Q-value로 사용해 행동 선택
                q_dist = self.policy_net(processed_state).exp()
                q_values = (q_dist * self.support).sum(2)
                return q_values.max(1)[1].view(1, 1)
            else:
                # 표준 DQN에서는 Q-value를 바로 사용
                return self.policy_net(processed_state).max(1)[1].view(1, 1)

    def select_action(self, state, eps_start, eps_end, eps_decay, evaluation_mode=False):
        """
        ### 설계 의도 (Design Intent)
        
        '탐험(Exploration)과 활용(Exploitation)의 딜레마'를 해결하기 위한 정책 결정 함수.
        - 탐험(Exploration): 더 나은 보상을 찾기 위해 새로운 행동을 시도하는 것.
        - 활용(Exploitation): 현재까지의 경험을 바탕으로 최선의 보상을 얻을 것으로 기대되는 행동을 선택하는 것.
        
        - Epsilon-Greedy 전략: 학습 초반에는 높은 `epsilon` 값으로 무작위 행동(탐험)을 자주 하여 경험을 쌓고,
          학습이 진행됨에 따라 `epsilon`을 점차 줄여 학습된 최적의 행동(활용)을 더 많이 선택.
          (수식: ε_t = ε_end + (ε_start - ε_end) * exp(-t / ε_decay))
          
        - Noisy Nets의 역할: `use_noisy` 활성화 시, Epsilon-Greedy 전략은 비활성화. 
          Noisy Nets는 신경망 가중치 자체에 노이즈를 주입하여 스스로 탐험의 정도를 학습하므로 별도 탐험 메커니즘 불필요.
          
        - 평가 모드: `evaluation_mode`에서는 `epsilon`과 관계없이 항상 탐욕적(greedy) 행동만 선택하여
          에이전트의 순수한 성능을 측정.
        """
        if evaluation_mode:
            return self._get_greedy_action(state)

        if self.use_noisy:
            return self._get_greedy_action(state)

        # --- Epsilon-Greedy 탐험 ---
        sample = random.random()
        self.eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        
        if sample > self.eps_threshold:
            # 활용(Exploitation): 현재까지 학습한 가장 좋은 행동을 선택
            return self._get_greedy_action(state)
        else:
            # 탐험(Exploration): 무작위로 행동을 선택하여 새로운 가능성 탐색
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def add_to_memory(self, state, action, reward, next_state, done):
        """
        ### 설계 의도 (Design Intent)
        
        에이전트가 겪은 하나의 '경험 조각(transition)'을 리플레이 버퍼에 저장.
        
        - PER의 특별 처리: `use_per` 활성화 시, 새로운 경험은 일단 가장 높은 우선순위(1.0)로 저장.
          이는 모든 새로운 경험이 적어도 한 번은 학습에 사용될 기회를 보장하기 위함. 
          실제 정확한 우선순위는 `optimize_model` 단계에서 TD-error 계산 후 업데이트.
        """
        if self.use_per:
            self.memory.add(1.0, state, action, reward, next_state, done)
        else:
            self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self):
        """
        ### 설계 의도 (Design Intent)
        
        DQN 학습의 핵심. 리플레이 버퍼에서 샘플링한 경험 배치로 신경망을 업데이트.
        '예측'과 '정답' 사이의 오차를 줄여나가는 과정.
        """
        if len(self.memory) < self.batch_size:
            return  # 버퍼에 충분한 데이터가 쌓일 때까지 학습 대기.

        # === 하위 목표 1: 경험 데이터 준비 (Prepare Experience Data) ===
        # 리플레이 버퍼에서 학습에 사용할 미니배치 샘플링.
        # PER 사용 시, '중요도 샘플링 가중치(is_weights)'가 함께 반환됨.
        if self.use_per:
            transitions, idxs, is_weights = self.memory.sample(self.batch_size)
            is_weights = torch.FloatTensor(is_weights).to(self.device)
        else:
            transitions = self.memory.sample(self.batch_size)
        
        batch = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))(*zip(*transitions))
        
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)

        if self.is_atari:
            state_batch = state_batch / 255.0
            next_state_batch = next_state_batch / 255.0

        # === 하위 목표 2: 현재 Q-value 예측 (Predict Current Q-values) ===
        # "현재 상태(s_t)에서 이 행동(a_t)을 했다면, Q-value는 얼마일까?"
        # policy_net을 사용하여 Q_θ(s_t, a_t) 계산.
        if self.use_distributional:
            log_q_distribution = self.policy_net(state_batch)
            log_q_s_a = log_q_distribution[range(self.batch_size), action_batch.squeeze(1)]
        else:
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # === 하위 목표 3: 타겟 Q-value 계산 (Compute Target Q-values) ===
        # "실제 보상(r)과 다음 상태의 최대 Q-value를 고려한 '정답(target)'은 얼마일까?"
        if self.use_distributional:
            with torch.no_grad():
                # --- 3a. 다음 상태의 행동 가치 분포 계산 ---
                next_log_q_distribution = self.target_net(next_state_batch)
                
                # --- 3b. Double DQN 로직 적용 ---
                # Double DQN: Q-value의 과대평가(overestimation) 문제를 완화하기 위해 행동 선택과 가치 평가를 분리하는 기법.
                # 왜 policy_net으로 행동을 선택하고 target_net으로 가치를 평가하는가?
                #   - 표준 DQN의 max 연산으로 인한 Q-value 과대평가 문제 완화.
                next_actions = self.policy_net(next_state_batch).exp()
                next_actions = (next_actions * self.support).sum(2).argmax(1)
                next_dist = next_log_q_distribution[range(self.batch_size), next_actions].exp()

                # --- 3c. 타겟 분포 프로젝션 (C51 알고리즘) ---
                # n-step 보상과 할인된 미래 가치 분포를 현재 시간으로 '투영(project)'.
                # (수식: Tz_j = r_t + (γ^n)z_j)
                projected_support = reward_batch.unsqueeze(1) + (self.gamma**self.n_steps) * self.support.unsqueeze(0) * (1 - done_batch.unsqueeze(1))
                projected_support = projected_support.clamp(self.v_min, self.v_max)

                # --- 확률 질량 분배 (BUG FIX) ---
                # 기존 로직은 b가 정수일 때(l==u), 확률 질량이 유실되는 버그가 있었음.
                # 아래는 l과 u가 같을 때와 다를 때 모두 올바르게 작동하는 수정된 로직.
                # upper_mass의 가중치 (b-l)은 b가 l에서 얼마나 떨어져 있는지를,
                # lower_mass의 가중치 (1-(b-l))은 b가 u에서 얼마나 떨어져 있는지를 나타냄 (l과 u의 간격은 1이므로).
                m = torch.zeros_like(next_dist, device=self.device)
                offset = torch.linspace(0, (self.batch_size - 1) * self.num_atoms, self.batch_size).long().unsqueeze(1).expand(self.batch_size, self.num_atoms).to(self.device)
                
                lower_mass = next_dist * (1.0 - (b - l.float()))
                upper_mass = next_dist * (b - l.float())

                m.view(-1).index_add_(0, (l + offset).view(-1), lower_mass.view(-1))
                m.view(-1).index_add_(0, (u + offset).view(-1), upper_mass.view(-1))
                
                target_q_distribution = m
        else: # Non-distributional DQN
            with torch.no_grad():
                if self.use_double:
                    # Double DQN: policy_net으로 최적 행동 선택, target_net으로 해당 행동의 가치 평가
                    # (수식: a'_t = argmax_a Q_θ(s_{t+1}, a), y_t = r_t + γ * Q_θ'(s_{t+1}, a'_t))
                    best_actions = self.policy_net(next_state_batch).argmax(1).unsqueeze(1)
                    next_state_values = self.target_net(next_state_batch).gather(1, best_actions).squeeze(1)
                else:
                    # 표준 DQN: target_net으로 최적 행동 선택 및 가치 평가
                    # (수식: y_t = r_t + γ * max_a' Q_θ'(s_{t+1}, a'))
                    next_state_values = self.target_net(next_state_batch).max(1)[0]
            
            next_state_values[done_batch] = 0.0 # 에피소드가 종료된 상태의 가치는 0
            # N-step 보상을 반영한 타겟 계산
            # (수식: y_t = G_{t:t+n} + (γ^n) * max_a' Q_θ'(s_{t+n}, a'))
            expected_state_action_values = (next_state_values * (self.gamma**self.n_steps)) + reward_batch

        # === 하위 목표 4: 손실 계산 (Compute Loss) ===
        # "예측값과 정답값의 차이(오차)는 얼마나 되는가?"
        if self.use_distributional:
            # 분포 DQN: 예측 분포와 타겟 분포 사이의 Cross-Entropy Loss 계산
            # (수식: L = -Σ_j m_j * log(p_j(s_t, a_t)))
            loss = - (target_q_distribution * log_q_s_a).sum(1)
        else:
            # 표준 DQN: Smooth L1 Loss (Huber Loss) 사용
            # (수식: L = E[(y_t - Q_θ(s_t, a_t))^2])
            criterion = nn.SmoothL1Loss(reduction='none' if self.use_per else 'mean')
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # --- PER을 위한 추가 처리 ---
        if self.use_per:
            # --- 4a. 버퍼 우선순위 업데이트 ---
            # TD-error(Temporal-Difference Error): 예측된 Q-value와 실제 관측된 타겟(정답) 사이의 차이. 에이전트의 '놀라움'의 정도.
            # 이 오차를 사용해 리플레이 버퍼의 우선순위 업데이트. 오차가 클수록 더 중요한 경험이므로 높은 우선순위 부여.
            errors = loss.detach().cpu().numpy() if self.use_distributional else torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
            for i in range(self.batch_size):
                self.memory.update(idxs[i], errors[i])
            
            # --- 4b. 중요도 샘플링 가중치 적용 ---
            # 중요도 샘플링(Importance Sampling): 특정 확률 분포(PER)에서 샘플링한 데이터로 다른 분포(균등분포)의 기댓값을 추정할 때, 샘플링 편향을 보정하기 위해 가중치를 사용하는 기법.
            # 왜 is_weights를 곱하는가?
            #   - PER은 중요한 샘플을 편향되게(biased) 많이 뽑으므로, 이 샘플들의 영향력을 그대로 반영하면 학습이 불안정해짐.
            #   - is_weights는 자주 뽑힌 샘플의 가중치를 낮추고 드물게 뽑힌 샘플의 가중치를 높여 편향을 보정.
            # (수식: Loss = E[w_i * δ_i^2], where w_i = (N * P(i))^(-β), δ_i는 TD-error)
            loss = (loss.squeeze(1) * is_weights).mean()

        # === 하위 목표 5: 모델 최적화 (Optimize Model) ===
        # 계산된 손실을 바탕으로 policy_net의 가중치를 업데이트 (역전파 및 경사 하강).
        self.optimizer.zero_grad()
        loss.backward()
        # 그래디언트 클리핑(Gradient Clipping): 그래디언트의 크기가 너무 커지지 않도록 일정 임계값으로 잘라내어, 학습 폭주를 막고 안정성을 높이는 기법.
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_net(self, tau):
        """
        ### 설계 의도 (Design Intent)
        
        학습 안정성을 위해 타겟 네트워크를 '부드럽게(softly)' 업데이트.
        
        - 왜 소프트 업데이트(Soft Update)를 사용하는가?
          - 하드 업데이트 (e.g., 10000 스텝마다 통째로 복사)는 타겟 값이 갑자기 크게 변해 학습을 불안정하게 만들 수 있음.
          - 소프트 업데이트는 매 스텝마다 정책망의 가중치를 아주 조금씩(τ만큼)만 타겟망에 반영, 
            타겟 값이 부드럽게 변하도록 만들어 학습 과정을 안정화.
        
        (수식: θ' ← τ*θ + (1-τ)*θ')
        """
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)
