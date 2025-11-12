import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """
    ### Noisy Net-A Factorized Gaussian Noise Layer
    
    #### 설계 의도 (Design Intent)
    - 학습 기반의 탐험(Learned Exploration): Epsilon-greedy와 같이 점차 감소하는 무작위 탐험이 아닌, 
      신경망 스스로 상태에 따라 탐험의 정도를 조절하도록 설계. 특정 상태에서 불확실성이 높다고 판단되면, 
      네트워크가 스스로 노이즈를 크게 만들어 더 다양한 행동을 탐색.
    - 파라미터 공간 노이즈(Parameter-space Noise): 행동 공간(action-space)에 직접 노이즈를 추가하는 대신,
      네트워크의 가중치(파라미터)에 노이즈를 주입. 이는 더 풍부하고 일관성 있는 탐험 정책을 생성.
      
    #### 핵심 아이디어 (Core Idea)
    - 선형 레이어의 가중치 `w`와 편향 `b`를 결정론적인 부분(`μ`)과 확률적인 부분(`σ * ε`)으로 분리.
      (수식: `w = μ_w + σ_w * ε_w`, `b = μ_b + σ_b * ε_b`)
    - `μ`와 `σ`는 일반적인 신경망 파라미터처럼 역전파를 통해 학습.
    - `ε`는 매 에피소드마다 다시 샘플링되는 고정된 노이즈.
    - **Factorised Gaussian Noise:** 계산 효율을 위해 노이즈 `ε`를 두 개의 작은 벡터 `ε_in`, `ε_out`의 외적(outer product)으로 생성.
    """
    def __init__(self, in_features, out_features, sigma_zero=0.5):
        """
        NoisyLinear 레이어 파라미터 초기화.
        
        Args:
            in_features (int): 입력 피처 수.
            out_features (int): 출력 피처 수.
            sigma_zero (float): 노이즈 표준편차 `σ`의 초기화 값.
        """
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero
        
        # 가중치와 편향의 평균(mu) 부분. 학습 가능한 파라미터.
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))

        # 가중치와 편향의 표준편차(sigma) 부분. 역시 학습 가능한 파라미터.
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # 노이즈(epsilon) 부분. 학습되지 않는 버퍼로 등록.
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """학습 가능한 파라미터(mu와 sigma)를 논문에 명시된 방식으로 초기화."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_zero / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_zero / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        """Factorised Gaussian Noise 생성을 위한 스케일링 함수. (sign(x) * sqrt(|x|))"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """매 에피소드 시작 또는 필요 시 새로운 노이즈를 샘플링."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # 두 노이즈 벡터의 외적(outer product)으로 전체 가중치 노이즈 생성
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        """
        Forward pass 정의.
        - 학습 시(self.training=True): 노이즈가 적용된 가중치를 사용하여 출력을 계산. 이는 탐험을 유도.
        - 평가 시(self.training=False): 노이즈를 제거하고 평균(mu) 가중치만 사용하여 결정론적(deterministic) 행동을 보장.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            return F.linear(x, weight, bias)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class QNetwork(nn.Module):
    """
    ### 표준 DQN을 위한 기본 Q-Network
    
    #### 설계 의도 (Design Intent)
    - 함수 근사(Function Approximation): 상태(state)를 입력으로 받아 각 행동(action)의 Q-value를 출력하는 함수를 근사.
    - 범용성: 상태가 벡터(vector) 형태로 주어지는 대부분의 표준 강화학습 환경(예: CartPole, Atari)에 적용 가능한
      기본적인 MLP(Multi-Layer Perceptron) 구조.
    """
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(QNetwork, self).__init__()
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        # Noisy Net 사용 여부에 따라 선형 레이어를 `nn.Linear` 또는 `NoisyLinear`로 동적 결정.
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        self.layer1 = LinearLayer(state_dim, 128)
        self.layer2 = LinearLayer(128, 128)
        # Distributional DQN의 경우, 각 행동마다 `num_atoms`개의 확률 값을 출력해야 하므로 출력 차원이 달라짐.
        self.layer3 = LinearLayer(128, action_dim * num_atoms if use_distributional else action_dim)

    def forward(self, input_tensor):
        """입력 상태로부터 Q-value 또는 Q-value 분포 계산."""
        x = F.relu(self.layer1(input_tensor))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        if self.use_distributional:
            # 출력을 (batch_size, action_dim, num_atoms) 형태로 재구성.
            x = x.view(-1, self.action_dim, self.num_atoms)
            # 각 행동의 분포에 대해 로그 소프트맥스를 취해 로그 확률(log-probabilities)로 변환.
            x = F.log_softmax(x, dim=2)
        
        return x

    def reset_noise(self):
        """네트워크 내의 모든 NoisyLinear 레이어의 노이즈를 리셋."""
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DuelingQNetwork(nn.Module):
    """
    ### Dueling DQN 아키텍처
    
    #### 설계 의도 (Design Intent)
    - 가치와 이점의 분리: Q-value를 '상태 자체의 가치(State Value, V(s))'와 '각 행동의 상대적 이점(Advantage, A(s,a))'으로 분리하여 학습.
    - 학습 효율 향상: 특정 상태에서는 어떤 행동을 하든 가치가 비슷할 수 있음. 이때, V(s)만 학습하면 되므로 불필요한 A(s,a) 학습을 줄여 효율을 높임.
      예를 들어, 눈앞에 장애물이 있어 무조건 충돌하는 상황이라면, 어떤 행동을 하든 상태 가치(V)는 낮고, 행동 간 이점(A) 차이는 거의 없음.
      
    #### 핵심 아이디어 (Core Idea)
    - Q-value를 두 개의 스트림(stream)으로 분리하여 계산 후 결합.
      - Value Stream: 상태가 얼마나 좋은지를 나타내는 V(s)를 예측.
      - Advantage Stream: 각 행동이 평균적인 행동보다 얼마나 더 나은지를 나타내는 A(s,a)를 예측.
    - 결합 공식: `Q(s, a) = V(s) + (A(s, a) - mean(A(s, a')))`
    - 왜 평균을 빼는가? (Identifiability): `Q = V + A` 수식만으로는 V와 A를 유일하게 결정할 수 없음 (e.g., V에 상수를 더하고 A에서 빼도 Q는 동일).
      Advantage의 평균을 0으로 만들어줌으로써 V가 상태 가치의 추정치 역할을 하도록 강제하여 학습을 안정화.
    """
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(DuelingQNetwork, self).__init__()
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        
        LinearLayer = NoisyLinear if use_noisy else nn.Linear

        # 공통 특징 추출 레이어
        self.feature_layer = nn.Sequential(LinearLayer(state_dim, 128), nn.ReLU())
        
        # 상태 가치(V(s)) 예측 스트림
        self.value_stream = nn.Sequential(LinearLayer(128, 128), nn.ReLU(), LinearLayer(128, num_atoms if use_distributional else 1))
        
        # 각 행동의 이점(A(s,a)) 예측 스트림
        self.advantage_stream = nn.Sequential(LinearLayer(128, 128), nn.ReLU(), LinearLayer(128, action_dim * num_atoms if use_distributional else action_dim))

    def forward(self, input_tensor):
        """Dueling 아키텍처 수식에 따라 Q-value 또는 Q-분포 계산."""
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
        """네트워크 내의 모든 NoisyLinear 레이어의 노이즈를 리셋."""
        for seq in [self.feature_layer, self.value_stream, self.advantage_stream]:
            for layer in seq:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

class DQN_CNN(nn.Module):
    """
    ### Atari 등 이미지 기반 환경을 위한 CNN 기반 Q-Network
    
    #### 설계 의도 (Design Intent)
    - 시각적 특징 추출: 고차원 데이터인 이미지(픽셀)로부터 의미 있는 시각적 특징(e.g., 공의 위치, 적의 형태)을 자동으로 추출.
    - 공간적 정보 활용: CNN(Convolutional Neural Network)을 사용하여 이미지의 공간적 구조(spatial structure)를 효과적으로 학습.
    
    #### 핵심 아이디어 (Core Idea)
    - Mnih et al. (2015) 아키텍처: "Human-level control through deep reinforcement learning" 논문에서 제안된 아키텍처를 기반으로 함.
      - 여러 개의 컨볼루션 레이어(Convolutional Layers)를 통해 이미지의 특징 맵(feature map)을 점진적으로 추출.
      - 추출된 특징 맵을 Flatten하여 1차원 벡터로 만들고, Fully-Connected Layers를 통과시켜 최종 Q-value를 계산.
    """
    def __init__(self, n_actions, use_noisy=False, use_distributional=False, num_atoms=51):
        super(DQN_CNN, self).__init__()
        self.use_distributional = use_distributional
        self.n_actions = n_actions
        self.num_atoms = num_atoms

        # 컨볼루션 레이어: 이미지 특징 추출.
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) # 입력 채널 4: Frame Stacking
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        # 완전 연결 레이어: 추출된 특징 바탕으로 Q-value 계산.
        # (64 * 7 * 7)은 conv3을 통과한 후의 특징 맵 크기.
        self.fc1 = LinearLayer(64 * 7 * 7, 512)
        self.fc2 = LinearLayer(512, n_actions * num_atoms if use_distributional else n_actions)

    def forward(self, x):
        """입력 이미지로부터 Q-value 또는 Q-value 분포 계산."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        if self.use_distributional:
            x = x.view(-1, self.n_actions, self.num_atoms)
            x = F.log_softmax(x, dim=2)
            
        return x

    def reset_noise(self):
        """네트워크 내의 모든 NoisyLinear 레이어의 노이즈를 리셋."""
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()