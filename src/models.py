import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    """Noisy Net-A Factorized Gaussian Noise Layer"""
    def __init__(self, in_features, out_features, sigma_zero=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_zero = sigma_zero
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_zero / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_zero / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            return F.linear(x, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            return F.linear(x, self.weight_mu, self.bias_mu)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(QNetwork, self).__init__()
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        if use_noisy:
            self.layer1 = NoisyLinear(state_dim, 128)
            self.layer2 = NoisyLinear(128, 128)
            self.layer3 = NoisyLinear(128, action_dim * num_atoms if use_distributional else action_dim)
        else:
            self.layer1 = nn.Linear(state_dim, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, action_dim * num_atoms if use_distributional else action_dim)

    def forward(self, input_tensor):
        x = F.relu(self.layer1(input_tensor))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)

        if self.use_distributional:
            x = x.view(-1, self.action_dim, self.num_atoms)
            x = F.log_softmax(x, dim=2)
        
        return x

    def reset_noise(self):
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(DuelingQNetwork, self).__init__()
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms

        if use_noisy:
            self.feature_layer = nn.Sequential(NoisyLinear(state_dim, 128), nn.ReLU())
            self.value_stream = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, num_atoms if use_distributional else 1))
            self.advantage_stream = nn.Sequential(NoisyLinear(128, 128), nn.ReLU(), NoisyLinear(128, action_dim * num_atoms if use_distributional else action_dim))
        else:
            self.feature_layer = nn.Sequential(nn.Linear(state_dim, 128), nn.ReLU())
            self.value_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, num_atoms if use_distributional else 1))
            self.advantage_stream = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, action_dim * num_atoms if use_distributional else action_dim))

    def forward(self, input_tensor):
        features = self.feature_layer(input_tensor)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        if self.use_distributional:
            advantage = advantage.view(-1, self.action_dim, self.num_atoms)
            value = value.view(-1, 1, self.num_atoms)
            q_distribution = value + (advantage - advantage.mean(dim=1, keepdim=True))
            output_tensor = F.log_softmax(q_distribution, dim=2)
        else:
            output_tensor = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return output_tensor

    def reset_noise(self):
        for seq in [self.feature_layer, self.value_stream, self.advantage_stream]:
            for layer in seq:
                if isinstance(layer, NoisyLinear):
                    layer.reset_noise()

class DQN_CNN(nn.Module):
    def __init__(self, n_actions, use_noisy=False, use_distributional=False, num_atoms=51):
        super(DQN_CNN, self).__init__()
        self.use_distributional = use_distributional
        self.n_actions = n_actions
        self.num_atoms = num_atoms

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        if use_noisy:
            self.fc1 = NoisyLinear(64 * 7 * 7, 512)
            self.fc2 = NoisyLinear(512, n_actions * num_atoms if use_distributional else n_actions)
        else:
            self.fc1 = nn.Linear(64 * 7 * 7, 512)
            self.fc2 = nn.Linear(512, n_actions * num_atoms if use_distributional else n_actions)

    def forward(self, x):
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
        for name, module in self.named_children():
            if isinstance(module, NoisyLinear):
                module.reset_noise()