import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    Noisy Net-A Factorized Gaussian Noise Layer.
    
    탐험(exploration)을 위해 가중치에 파라메트릭 노이즈를 추가하는 선형 레이어 구현.
    Epsilon-greedy 방식 대신, 학습 가능한 노이즈를 통해 더 정교한 탐험 수행.
    논문 'Noisy Networks for Exploration'에 제안된 Factorised Gaussian Noise 방식 적용.
    """
    def __init__(self, in_features, out_features, sigma_zero=0.5):
        """
        NoisyLinear 레이어 파라미터 초기화.
        
        Args:
            in_features (int): 입력 피처 수.
            out_features (int): 출력 피처 수.
            sigma_zero (float): 노이즈 표준편차 초기화 값.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero
        
        # 가중치와 편향의 평균(mu)은 학습 가능한 파라미터.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # 가중치와 편향의 표준편차(sigma) 또한 학습 가능한 파라미터.
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # 노이즈(epsilon)는 학습되지 않는 버퍼로 등록.
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        학습 가능한 파라미터(mu와 sigma)를 논문에 따라 초기화.
        """
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_zero / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_zero / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """
        Factorised Gaussian Noise 생성을 위한 스케일링 함수.
        sign(x) * sqrt(|x|) 변환 적용.
        """
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """
        매 에피소드 시작 시 새로운 노이즈 샘플링.
        """
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        Forward pass 정의.
        학습 시에는 노이즈 적용 가중치 사용, 평가 시에는 노이즈 없는 평균 가중치만 사용.
        """
        if self.training:
            # w = μ_w + σ_w * ε_w, b = μ_b + σ_b * ε_b
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            # 평가 시에는 결정론적(deterministic) 행동을 위해 평균 가중치만 사용.
            return F.linear(x, self.weight_mu, self.bias_mu)

class QNetwork(nn.Module):
    """
    표준 DQN을 위한 기본 Q-Network.
    상태가 벡터(vector) 형태로 주어지는 환경(예: CartPole)을 위한 MLP(Multi-Layer Perceptron) 구조.
    """
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(QNetwork, self).__init__()
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        # Noisy Net 사용 여부에 따라 선형 레이어를 nn.Linear 또는 NoisyLinear로 결정.
        if use_noisy:
            self.layer1 = NoisyLinear(state_dim, 128)
            self.layer2 = NoisyLinear(128, 128)
            self.layer3 = NoisyLinear(128, action_dim * num_atoms if use_distributional else action_dim)
        else:
            self.layer1 = nn.Linear(state_dim, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, action_dim * num_atoms if use_distributional else action_dim)

    def forward(self, input_tensor):
        """
        입력 상태로부터 Q-value 또는 Q-value 분포 계산.
        """
        x = F.relu(self.layer1(input_tensor))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        # Distributional DQN의 경우, 출력을 행동 차원과 아톰 차원으로 재구성하고 로그 확률로 변환.
        if self.use_distributional:
            x = x.view(-1, self.action_dim, self.num_atoms)
            x = F.log_softmax(x, dim=2)
        
        return x

    def reset_noise(self):
        """
        모든 NoisyLinear 레이어의 노이즈 리셋.
        """
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DuelingQNetwork(nn.Module):
    """
    Dueling DQN 아키텍처 구현 Q-Network.
    상태 가치(State Value) 스트림과 행동 이점(Advantage) 스트림을 분리하여 학습 효율 향상.
    """
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(DuelingQNetwork, self).__init__()
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        # 공통 특징 추출 레이어
        if use_noisy:
            self.feature_layer = nn.Sequential(NoisyLinear(state_dim, 128), nn.ReLU())
            # 상태 가치(V(s)) 예측 스트림
            self.value_stream = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, num_atoms if use_distributional else 1))
            # 각 행동의 이점(A(s,a)) 예측 스트림
            self.advantage_stream = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, action_dim * num_atoms if use_distributional else action_dim))
        else:
            self.feature_layer = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
            self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_atoms if use_distributional else 1))
            self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim * num_atoms if use_distributional else action_dim))

    def forward(self, input_tensor):
        """
        Dueling 아키텍처 수식에 따라 Q-value 계산.
        Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))
        """
        features = self.feature_layer(input_tensor)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        if self.use_distributional:
            # Distributional의 경우, 각 스트림 출력을 분포 형태로 다룸.
            advantage = advantage.view(-1, self.action_dim, self.num_atoms)
            value = value.view(-1, 1, self.num_atoms)
            # 수식에 따라 Q-value 분포 결합.
            q_distribution = value + (advantage - advantage.mean(dim=1, keepdim=True))
            output_tensor = F.log_softmax(q_distribution, dim=2)
        else:
            # 표준 Dueling의 경우, 스칼라 값으로 Q-value 결합.
            output_tensor = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return output_tensor

    def reset_noise(self):
        """
        모든 NoisyLinear 레이어의 노이즈 리셋.
        """
        for seq in [self.feature_layer, self.value_stream, self.advantage_stream]:
            for layer in seq:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

class DQN_CNN(nn.Module):
    """
    Atari 등 이미지 기반 환경을 위한 CNN 기반 Q-Network.
    Mnih et al. (2015) 논문 제안 아키텍처 적용.
    """
    def __init__(self, n_actions, use_noisy=False, use_distributional=False, num_atoms=51):
        super(DQN_CNN, self).__init__()
        self.use_distributional = use_distributional
        self.n_actions = n_actions
        self.num_atoms = num_atoms

        # 컨볼루션 레이어: 이미지 특징 추출.
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) # 4 프레임 스택 입력
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # 완전 연결 레이어: 추출된 특징 바탕으로 Q-value 계산.
        if use_noisy:
            self.fc1 = NoisyLinear(64 * 7 * 7, 512)
            self.fc2 = NoisyLinear(512, n_actions * num_atoms if use_distributional else n_actions)
        else:
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, n_actions * num_atoms if use_distributional else n_actions)

    def forward(self, x):
        """
        입력 이미지로부터 Q-value 또는 Q-value 분포 계산.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.use_distributional:
            x = x.view(-1, self.n_actions, self.num_atoms)
            x = F.log_softmax(x, dim=2)
            
        return x

    def reset_noise(self):
        """
        모든 NoisyLinear 레이어의 노이즈 리셋.
        """
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()