import random
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- Standard Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# --- Prioritized Replay Buffer ---
# SumTree for efficient sampling
class SumTree:
    write_index = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.transitions_data = np.zeros(capacity, dtype=object)
        self.number_of_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, sample_value):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if sample_value <= self.tree[left]:
            return self._retrieve(left, sample_value)
        else:
            return self._retrieve(right, sample_value - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, priority, data):
        idx = self.write_index + self.capacity - 1

        self.transitions_data[self.write_index] = data
        self.update(idx, priority)

        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0

        if self.number_of_entries < self.capacity:
            self.number_of_entries += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, sample_value):
        idx = self._retrieve(0, sample_value)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.transitions_data[dataIdx])


class PrioritizedReplayBuffer:
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, *args):
        priority = self._get_priority(error)
        self.tree.add(priority, Transition(*args))

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            segment_start = segment * i
            segment_end = segment * (i + 1)
            sample_value = random.uniform(segment_start, segment_end)
            (idx, priority, data) = self.tree.get(sample_value)
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        importance_sampling_weight = np.power(self.tree.number_of_entries * sampling_probabilities, -self.beta)
        importance_sampling_weight /= importance_sampling_weight.max()

        return batch, idxs, importance_sampling_weight

    def update(self, idx, error):
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def __len__(self):
        return self.tree.number_of_entries