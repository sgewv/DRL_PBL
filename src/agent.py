import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .models import QNetwork, DuelingQNetwork, DQN_CNN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, Transition

class DQNAgent:
    def __init__(self, state_dim, action_dim, is_atari=False, use_dueling=False, use_double=False, use_per=False, use_noisy=False, use_distributional=False, num_atoms=51, v_min=-10, v_max=10, learning_rate=1e-4):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_atari = is_atari
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_dueling = use_dueling
        self.use_double = use_double
        self.use_per = use_per
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        if is_atari:
            # For Atari, Dueling is often handled differently or not used in simple CNNs
            QNetworkModel = DQN_CNN
            self.policy_net = QNetworkModel(action_dim, use_noisy=use_noisy, use_distributional=use_distributional, num_atoms=num_atoms).to(self.device)
            self.target_net = QNetworkModel(action_dim, use_noisy=use_noisy, use_distributional=use_distributional, num_atoms=num_atoms).to(self.device)
        else:
            QNetworkModel = DuelingQNetwork if use_dueling else QNetwork
            self.policy_net = QNetworkModel(state_dim, action_dim, use_noisy=use_noisy, use_distributional=use_distributional, num_atoms=num_atoms, v_min=v_min, v_max=v_max).to(self.device)
            self.target_net = QNetworkModel(state_dim, action_dim, use_noisy=use_noisy, use_distributional=use_distributional, num_atoms=num_atoms, v_min=v_min, v_max=v_max).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=learning_rate, amsgrad=True)
        
        buffer_capacity = 100000 if is_atari else 10000
        if use_per:
            self.memory = PrioritizedReplayBuffer(buffer_capacity)
        else:
            self.memory = ReplayBuffer(buffer_capacity)

        self.steps_done = 0

        if self.use_distributional:
            self.support = torch.linspace(v_min, v_max, num_atoms).to(self.device)
            self.delta_z = (v_max - v_min) / (num_atoms - 1)

    def _get_greedy_action(self, state):
        """Selects an action greedily from the policy network."""
        with torch.no_grad():
            # Normalize state if it's an Atari frame
            processed_state = state / 255.0 if self.is_atari else state
            if self.use_distributional:
                q_dist = self.policy_net(processed_state).exp()
                q_values = (q_dist * self.support).sum(2)
                return q_values.max(1)[1].view(1, 1)
            else:
                return self.policy_net(processed_state).max(1)[1].view(1, 1)

    def select_action(self, state, eps_start, eps_end, eps_decay, evaluation_mode=False):
        if evaluation_mode:
            return self._get_greedy_action(state)

        if self.use_noisy:
            return self._get_greedy_action(state)

        sample = random.random()
        self.eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * self.steps_done / eps_decay)
        self.steps_done += 1
        if sample > self.eps_threshold:
            return self._get_greedy_action(state)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)

    def add_to_memory(self, state, action, reward, next_state, done):
        if self.use_per:
            self.memory.add(1.0, state, action, reward, next_state, done) # Give max priority
        else:
            self.memory.push(state, action, reward, next_state, done)

    def optimize_model(self, batch_size, gamma, n_steps):
        if len(self.memory) < batch_size:
            return None # Return None if not optimizing
        
        # For Atari, a smaller batch size is common
        if self.is_atari:
            batch_size = 32
        
        if len(self.memory) < batch_size:
            return None

        if self.use_per:
            transitions, idxs, is_weights = self.memory.sample(batch_size)
        else:
            transitions = self.memory.sample(batch_size)
        
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Normalize pixel values for Atari
        if self.is_atari:
            state_batch = state_batch / 255.0
            non_final_next_states = non_final_next_states / 255.0

        if self.use_distributional:
            # (Distributional logic remains largely the same, but the reward_batch is now an n-step return)
            log_q_distribution = self.policy_net(state_batch)[range(batch_size), action_batch.squeeze(1)]

            with torch.no_grad():
                target_q_distribution = torch.zeros(batch_size, self.num_atoms, device=self.device)
                
                if non_final_mask.any():
                    next_log_dist = self.target_net(non_final_next_states)
                    next_dist = next_log_dist.exp()
                    
                    if self.use_double:
                        next_q_policy = (self.policy_net(non_final_next_states).exp() * self.support).sum(2)
                        next_actions = next_q_policy.argmax(1)
                    else:
                        next_q = (next_dist * self.support).sum(2)
                        next_actions = next_q.argmax(1)
                    
                    next_dist = next_dist[range(len(non_final_next_states)), next_actions]

                    # N-step gamma discount applied here
                    projected_support = reward_batch[non_final_mask].unsqueeze(1) + (gamma**n_steps) * self.support.unsqueeze(0)
                    projected_support = projected_support.clamp(self.v_min, self.v_max)

                    projected_index = (projected_support - self.v_min) / self.delta_z
                    lower_index = projected_index.floor().long()
                    upper_index = projected_index.ceil().long()
                    
                    l_weight = next_dist * (upper_index.float() - projected_index)
                    u_weight = next_dist * (projected_index - lower_index.float())
                    
                    proj_dist = torch.zeros_like(next_dist, device=self.device)
                    proj_dist.scatter_add_(1, lower_index, l_weight)
                    proj_dist.scatter_add_(1, upper_index, u_weight)
                    target_q_distribution[non_final_mask] = proj_dist

                final_mask = ~non_final_mask
                if final_mask.any():
                    projected_support = reward_batch[final_mask].clamp(self.v_min, self.v_max)
                    projected_index = (projected_support - self.v_min) / self.delta_z
                    lower_index = projected_index.floor().long()
                    upper_index = projected_index.ceil().long()
                    
                    lower_index.clamp_(0, self.num_atoms - 1)
                    upper_index.clamp_(0, self.num_atoms - 1)

                    l_weight = 1.0 - (projected_index - lower_index.float())
                    u_weight = projected_index - lower_index.float()

                    proj_dist_final = torch.zeros(final_mask.sum(), self.num_atoms, device=self.device)
                    proj_dist_final.scatter_add_(1, lower_index.unsqueeze(1), l_weight.unsqueeze(1))
                    proj_dist_final.scatter_add_(1, upper_index.unsqueeze(1), u_weight.unsqueeze(1))
                    target_q_distribution[final_mask] = proj_dist_final

            kl_divergence = - (target_q_distribution * log_q_distribution).sum(1)
            
            if self.use_per:
                errors = kl_divergence.detach().cpu().numpy()
                for i in range(batch_size):
                    self.memory.update(idxs[i], errors[i])
                loss = (kl_divergence * torch.FloatTensor(is_weights).to(self.device)).mean()
            else:
                loss = kl_divergence.mean()

        else: # Standard (non-distributional) DQN
            state_action_values = self.policy_net(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(batch_size, device=self.device)
            with torch.no_grad():
                if self.use_double:
                    best_actions = self.policy_net(non_final_next_states).argmax(1).unsqueeze(1)
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, best_actions).squeeze(1)
                else:
                    next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]
            
            # Compute the expected Q values using n-step logic
            expected_state_action_values = (next_state_values * (gamma**n_steps)) + reward_batch

            if self.use_per:
                errors = torch.abs(state_action_values - expected_state_action_values.unsqueeze(1)).detach().cpu().numpy()
                for i in range(batch_size):
                    self.memory.update(idxs[i], errors[i])
                criterion = nn.SmoothL1Loss(reduction='none')
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
                is_weights = torch.FloatTensor(is_weights).to(self.device)
                loss = (loss * is_weights.unsqueeze(1)).mean()
            else:
                criterion = nn.SmoothL1Loss()
                loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Log the loss to W&B
        import wandb
        wandb.log({"loss": loss.item()})
        return loss.item()

    def update_target_net(self, tau=0.005):
        # For Atari, a hard update is sometimes used. Here we stick with soft updates.
        # if self.is_atari and self.steps_done % 10000 == 0: # Hard update every C steps
        #     self.target_net.load_state_dict(self.policy_net.state_dict())
        #     return
        
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key]*(1-tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def reset_noise(self):
        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
