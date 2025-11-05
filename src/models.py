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
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        LinearLayer = NoisyLinear if use_noisy else nn.Linear

        self.layer1 = LinearLayer(state_dim, 128)
        self.layer2 = LinearLayer(128, 128)
        
        if use_distributional:
            self.layer3 = LinearLayer(128, action_dim * num_atoms)
            self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))
        else:
            self.layer3 = LinearLayer(128, action_dim)

    def forward(self, input_tensor):
        input_tensor = F.relu(self.layer1(input_tensor))
        input_tensor = F.relu(self.layer2(input_tensor))
        input_tensor = self.layer3(input_tensor)

        if self.use_distributional:
            input_tensor = input_tensor.view(-1, self.action_dim, self.num_atoms)
            input_tensor = F.log_softmax(input_tensor, dim=2)
        
        print(f"[QNetwork] Output shape: {input_tensor.shape}")
        return input_tensor

    def reset_noise(self):
        if self.use_noisy:
            for name, module in self.named_children():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10):
        super(DuelingQNetwork, self).__init__()
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.action_dim = action_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        LinearLayer = NoisyLinear if use_noisy else nn.Linear

        self.feature_layer = nn.Sequential(
            LinearLayer(state_dim, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            LinearLayer(128, 128),
            nn.ReLU(),
            LinearLayer(128, num_atoms if use_distributional else 1)
        )

        self.advantage_stream = nn.Sequential(
            LinearLayer(128, 128),
            nn.ReLU(),
            LinearLayer(128, action_dim * num_atoms if use_distributional else action_dim)
        )

        if use_distributional:
            self.register_buffer('support', torch.linspace(v_min, v_max, num_atoms))

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
        
        print(f"[DuelingQNetwork] Output shape: {output_tensor.shape}")
        return output_tensor

    def reset_noise(self):
        if self.use_noisy:
            for name, module in self.named_children():
                if hasattr(module, 'reset_noise'):
                    module.reset_noise()
                elif isinstance(module, NoisyLinear):
                    module.reset_noise()
            for seq_name, sequential in [('feature_layer', self.feature_layer), ('value_stream', self.value_stream), ('advantage_stream', self.advantage_stream)]:
                for name, module in sequential.named_children():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()