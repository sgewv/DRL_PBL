import random
import numpy as np
from collections import namedtuple, deque

# (상태, 행동, 보상, 다음 상태, 종료 여부)를 하나의 트랜지션(Transition)으로 묶어주는 튜플.
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- 표준 경험 리플레이 버퍼 (Standard Experience Replay Buffer) ---
class ReplayBuffer:
    """
    ### 설계 의도 (Design Intent)
    
    에이전트의 경험을 저장하고 재사용하기 위한 메모리 구조.
    
    - **데이터 효율성(Data Efficiency):** 에이전트가 한 번 겪은 경험(transition)을 여러 번 학습에 재사용하여 데이터 효율을 높임.
    - **학습 안정성(Learning Stability):** 경험을 시간 순서대로 학습하면 샘플 간 상관관계(correlation)가 높아 학습이 불안정해짐.
      버퍼에서 무작위로 샘플링하여 이 상관관계를 깨뜨리고, i.i.d(independent and identically distributed) 가정을 만족시켜 학습을 안정화.
      
    #### 핵심 아이디어 (Core Idea)
    - `deque` 자료구조에 `maxlen`을 설정하여, 버퍼가 가득 차면 가장 오래된 경험부터 자동으로 삭제되는 FIFO(First-In, First-Out) 방식의 순환 버퍼 구현.
    """
    def __init__(self, capacity):
        """
        버퍼 초기화.
        :param capacity: 버퍼가 저장할 수 있는 최대 경험의 수.
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """하나의 트랜지션(경험)을 버퍼에 저장."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """버퍼에서 `batch_size` 만큼의 경험을 무작위로 균등하게 샘플링."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """버퍼에 현재 저장된 경험의 수를 반환."""
        return len(self.memory)


# --- 우선순위 경험 리플레이 (Prioritized Experience Replay, PER) ---
class SumTree:
    """
    ### SumTree 자료구조
    
    #### 설계 의도 (Design Intent)
    - **효율적인 우선순위 기반 샘플링:** PER을 위해, 각 경험의 우선순위(priority)에 비례하여 O(log N)의 시간 복잡도로 효율적인 샘플링을 지원.
    
    #### 핵심 아이디어 (Core Idea)
    - **이진 트리 구조:**
      - **리프 노드(Leaf Nodes):** 각 경험의 우선순위(priority) 값을 저장.
      - **내부 노드(Internal Nodes):** 두 자식 노드의 우선순위 합을 저장.
      - **루트 노드(Root Node):** 모든 리프 노드의 우선순위 총합(`total_priority`)을 저장.
    - **샘플링 과정:**
      1. `[0, total_priority]` 범위에서 무작위 값 `s`를 샘플링.
      2. 루트 노드부터 시작하여 `s` 값에 해당하는 리프 노드를 재귀적으로 탐색.
      3. `s`가 왼쪽 자식 노드의 값보다 작으면 왼쪽으로, 크면 오른쪽으로 이동 (이때 `s`에서 왼쪽 자식 값을 뺌).
    """
    write_index = 0  # 데이터를 쓸 다음 인덱스 (순환 큐처럼 동작)

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # 트리 구조를 위한 배열
        self.transitions_data = np.zeros(capacity, dtype=object)  # 실제 경험 데이터를 저장하는 배열
        self.number_of_entries = 0

    def _propagate(self, idx, change):
        """리프 노드의 우선순위 변경(`change`)을 루트까지 전파하여 부모 노드들의 합을 갱신."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, sample_value):
        """루트부터 시작하여 `sample_value`에 해당하는 리프 노드를 O(log N)으로 검색."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree): # 리프 노드에 도달
            return idx

        if sample_value <= self.tree[left]:
            return self._retrieve(left, sample_value)
        else:
            return self._retrieve(right, sample_value - self.tree[left])

    def total(self):
        """모든 우선순위의 총합 (루트 노드의 값) 반환."""
        return self.tree[0]

    def add(self, priority, data):
        """새로운 데이터와 그 우선순위를 트리에 추가."""
        idx = self.write_index + self.capacity - 1

        self.transitions_data[self.write_index] = data
        self.update(idx, priority)

        self.write_index += 1
        if self.write_index >= self.capacity:
            self.write_index = 0

        if self.number_of_entries < self.capacity:
            self.number_of_entries += 1

    def update(self, idx, priority):
        """특정 리프 노드(`idx`)의 우선순위를 `priority` 값으로 갱신."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, sample_value):
        """`sample_value`에 해당하는 (트리 인덱스, 우선순위, 데이터)를 반환."""
        idx = self._retrieve(0, sample_value)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.transitions_data[dataIdx])


class PrioritizedReplayBuffer:
    """
    ### 우선순위 경험 리플레이 (PER) 버퍼
    
    #### 설계 의도 (Design Intent)
    - 효율적인 학습: 모든 경험을 동일한 가치로 취급하는 대신, 학습에 더 도움이 될 '중요한' 경험을 더 자주 샘플링.
    - '놀라움' 기반 학습: 예측이 크게 빗나간 경험, 즉 TD-error가 큰 경험일수록 '놀라움'이 크고 배울 점이 많다고 가정.
    
    #### 핵심 아이디어 (Core Idea)
    - 우선순위 계산: 각 경험의 우선순위 `p_i`는 TD-error `δ_i`의 절댓값에 기반하여 계산.
      (수식: `p_i = (|δ_i| + ε)^α`)
      - `α` (alpha): 우선순위화 정도를 조절 (0: 균등 샘플링, 1: TD-error에 정비례).
      - `ε` (epsilon): TD-error가 0인 경험도 샘플링될 최소한의 확률을 보장.
    - 편향 보정 (Bias Correction): 우선순위에 따라 편향되게 샘플링하면, 특정 경험만 과도하게 학습하여 학습이 불안정해질 수 있음.
      중요도 샘플링(Importance Sampling, IS) 가중치를 사용하여 이 편향을 보정.
      (수식: `w_i = (N * P(i))^(-β)`)
      - `β` (beta): 편향 보정의 강도를 조절. 학습 초기에는 작은 값에서 시작하여 점차 1로 증가(annealing)시켜, 학습 후반으로 갈수록 강하게 보정.
      - 자주 뽑히는 샘플(P(i)가 높음)은 낮은 가중치(`w_i`)를, 드물게 뽑히는 샘플은 높은 가중치를 부여하여 손실(loss)에 반영.
    """
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        """TD-Error를 입력받아 실제 우선순위 값을 계산."""
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, *args):
        """새로운 경험과 TD-Error를 버퍼에 추가."""
        priority = self._get_priority(error)
        self.tree.add(priority, Transition(*args))

    def sample(self, batch_size):
        """우선순위에 비례하여 경험을 샘플링하고, 중요도 샘플링(IS) 가중치를 계산."""
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size
        priorities = []

        # beta 값을 0.4에서 1.0까지 점진적으로 증가(annealing).
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            # --- 층화 샘플링 (Stratified Sampling) ---
            # 전체 우선순위 합을 `batch_size`개의 구간(segment)으로 나누고, 각 구간에서 하나씩 샘플링.
            # 이를 통해 샘플링의 다양성을 확보하고, 우선순위가 낮은 경험도 샘플링될 기회를 보장.
            segment_start = segment * i
            segment_end = segment * (i + 1)
            sample_value = random.uniform(segment_start, segment_end)
            
            (idx, priority, data) = self.tree.get(sample_value)
            
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        # --- 중요도 샘플링(IS) 가중치 계산 ---
        sampling_probabilities = priorities / self.tree.total()
        
        # IS 가중치 w_i = (N * P(i))^(-beta)
        importance_sampling_weight = np.power(self.tree.number_of_entries * sampling_probabilities, -self.beta)
        
        # 안정적인 학습을 위해 가중치를 최대 가중치로 정규화.
        importance_sampling_weight /= importance_sampling_weight.max()

        return batch, idxs, importance_sampling_weight

    def update(self, idx, error):
        """학습 후, 샘플링되었던 경험의 TD-Error를 새로운 값으로 갱신."""
        priority = self._get_priority(error)
        self.tree.update(idx, priority)

    def __len__(self):
        """버퍼에 현재 저장된 경험의 수를 반환."""
        return self.tree.number_of_entries