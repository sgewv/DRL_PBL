import random
import numpy as np
from collections import namedtuple, deque

# (상태, 행동, 보상, 다음 상태, 종료 여부)를 하나의 객체로 묶음
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

# --- 표준 경험 리플레이 버퍼 (Standard Replay Buffer) ---
class ReplayBuffer:
    """
    DQN을 위한 표준 경험 리플레이 버퍼
    데이터를 저장하고(push), 무작위로 샘플링(sample)
    """
    def __init__(self, capacity):
        """
        버퍼 초기화합
        :param capacity: 버퍼가 저장할 수 있는 최대 경험의 수
        """
        # deque는 양방향 큐 자료구조로, maxlen이 설정되면
        # 버퍼가 가득 찼을 때 가장 오래된 데이터부터 자동으로 삭제
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """트랜지션(경험)을 버퍼에 저장"""
        # (state, action, reward, next_state, done)을 Transition 객체로 만들어 메모리에 추가
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """버퍼에서 batch_size만큼의 경험을 무작위로 샘플링"""
        # random.sample을 사용하여 메모리에 저장된 모든 경험 중 균일한 확률로 추출
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """버퍼에 현재 저장된 경험의 수를 반환"""
        return len(self.memory)


# --- 우선순위 경험 리플레이 버퍼 (Prioritized Replay Buffer) ---
class SumTree:
    """
    PER을 효율적으로 구현하기 위한 SumTree 자료구조
    이진 트리 구조로, 리프 노드에는 각 경험의 '우선순위(priority)'를 저장하고,
    부모 노드는 자식 노드들의 합을 저장
    루트 노드는 모든 우선순위의 총합

    우선순위에 비례하는 샘플링을 O(log N) 시간 복잡도로 수행
    """
    write_index = 0  # 데이터를 쓸 다음 인덱스 (순환 큐처럼 동작)

    def __init__(self, capacity):
        """
        SumTree를 초기화
        :param capacity: 버퍼의 최대 용량 (리프 노드의 수)
        """
        self.capacity = capacity
        # 트리의 전체 노드 수는 (2 * capacity - 1)개
        self.tree = np.zeros(2 * capacity - 1)
        # 실제 트랜지션 데이터(Transition 객체)를 저장하는 배열
        self.transitions_data = np.zeros(capacity, dtype=object)
        self.number_of_entries = 0  # 현재 저장된 데이터의 수

    def _propagate(self, idx, change):
        """
        특정 리프 노드(idx)의 값이 'change'만큼 변경되었을 때,
        이 변경 사항을 루트 노드까지 전파(propagate)하여 부모 노드들의 합을 갱신
        """
        parent = (idx - 1) // 2  # 부모 노드의 인덱스
        self.tree[parent] += change  # 부모 노드에 변경 사항 반영
        if parent != 0:
            # 루트 노드(0)에 도달할 때까지 재귀적으로 호출
            self._propagate(parent, change)

    def _retrieve(self, idx, sample_value):
        """
        루트 노드(idx=0)부터 시작하여, 'sample_value'에 해당하는
        우선순위를 가진 리프 노드를 O(log N)으로 검색
        (sample_value: 0 ~ total_priority 사이의 랜덤 값)
        """
        left = 2 * idx + 1    # 왼쪽 자식 노드 인덱스
        right = left + 1      # 오른쪽 자식 노드 인덱스

        if left >= len(self.tree):
            # 자식이 없는 리프 노드에 도달하면 해당 인덱스 반환
            return idx

        if sample_value <= self.tree[left]:
            # sample_value가 왼쪽 자식 노드의 합보다 작거나 같으면 왼쪽으로 계속 탐색
            return self._retrieve(left, sample_value)
        else:
            # sample_value가 왼쪽 자식의 합보다 크면 오른쪽으로 탐색
            # 이 때, sample_value에서 왼쪽 자식의 합을 빼고 탐색을 계속합니다.
            return self._retrieve(right, sample_value - self.tree[left])

    def total(self):
        """모든 우선순위의 총합을 반환. (루트 노드의 값)"""
        return self.tree[0]

    def add(self, priority, data):
        """
        새로운 데이터(data)와 그 우선순위(priority)를 트리에 추가
        """
        # 리프 노드에 쓸 인덱스를 계산 (트리 배열의 뒷부분이 리프 노드)
        idx = self.write_index + self.capacity - 1

        self.transitions_data[self.write_index] = data  # 실제 데이터 저장
        self.update(idx, priority)  # 트리에 우선순위 값을 갱신 (및 전파)

        self.write_index += 1  # 다음 쓸 위치로 인덱스 이동
        if self.write_index >= self.capacity:
            # 인덱스가 용량을 초과하면 0으로 리셋 (순환 큐)
            self.write_index = 0

        if self.number_of_entries < self.capacity:
            # 저장된 데이터 수 증가 (용량 한도 내에서)
            self.number_of_entries += 1

    def update(self, idx, priority):
        """
        트리의 특정 리프 노드(idx)의 우선순위를 'priority' 값으로 갱신
        """
        change = priority - self.tree[idx]  # 기존 값과의 차이(change) 계산
        self.tree[idx] = priority           # 새 우선순위로 값 변경
        self._propagate(idx, change)        # 루트까지 변경 사항 전파

    def get(self, sample_value):
        """
        'sample_value'에 해당하는 (트리 인덱스, 우선순위, 데이터)를 반환
        """
        idx = self._retrieve(0, sample_value)      # sample_value로 트리 인덱스 검색
        dataIdx = idx - self.capacity + 1          # 실제 데이터 배열의 인덱스 계산
        return (idx, self.tree[idx], self.transitions_data[dataIdx])


class PrioritizedReplayBuffer:
    """
    우선순위 경험 리플레이(PER) 버퍼
    SumTree를 사용하여 TD-Error(놀라움)가 큰 경험을 더 자주 샘플링
    또한, 이로 인한 편향(bias)을 중요도 샘플링(Importance Sampling)으로 보정
    """
    # PER 하이퍼파라미터
    e = 0.01  # epsilon: TD-Error가 0일 때도 최소한의 우선순위를 보장하기 위한 작은 값 (p = (|TD-Error| + e)^a)
    a = 0.6   # alpha: 우선순위화 지수 (0~1). 0이면 균등 샘플링, 1이면 TD-Error에 정비례.
    beta = 0.4  # beta: 중요도 샘플링(IS) 가중치 지수 (0~1). 편향 보정의 강도를 조절.
    beta_increment_per_sampling = 0.001 # beta를 0.4에서 1.0까지 점진적으로 증가시키기 위한 값

    def __init__(self, capacity):
        """버퍼 초기화. 내부적으로 SumTree를 생성"""
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        """
        TD-Error(error)를 입력받아 실제 우선순위(priority) 값을 계산
        공식: p = (|TD-Error| + e)^a
        """
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, *args):
        """
        새로운 경험(*args)과 해당 경험의 TD-Error(error)를 버퍼에 추가
        """
        priority = self._get_priority(error)  # TD-Error로 우선순위 계산
        self.tree.add(priority, Transition(*args))  # SumTree에 (우선순위, 데이터) 추가

    def sample(self, batch_size):
        """
        우선순위에 비례하여 batch_size만큼의 경험을 샘플링
        또한, 중요도 샘플링(IS) 가중치를 계산하여 반환
        """
        batch = []  # 샘플링된 데이터(Transition 객체)를 담을 리스트
        idxs = []   # 샘플링된 데이터의 트리 인덱스를 담을 리스트 (나중에 update 시 필요)
        segment = self.tree.total() / batch_size  # 전체 우선순위 합을 배치 크기로 나눔 (층화 샘플링)
        priorities = []  # 샘플링된 데이터의 우선순위를 담을 리스트

        # beta 값을 0.4에서 1.0까지 점진적으로 증가(annealing)
        # 학습 초반에는 편향 보정을 약하게, 후반으로 갈수록 강하게 적용
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(batch_size):
            # 층화 샘플링(Stratified Sampling)
            # i번째 세그먼트(구간) [segment * i, segment * (i+1)] 내에서 랜덤 값을 뽑음.
            # 이렇게 하면 우선순위가 낮은 경험도 샘플링될 기회를 보장받음.
            segment_start = segment * i
            segment_end = segment * (i + 1)
            sample_value = random.uniform(segment_start, segment_end)
            
            # SumTree.get을 이용해 sample_value에 해당하는 (인덱스, 우선순위, 데이터)를 가져옴
            (idx, priority, data) = self.tree.get(sample_value)
            
            priorities.append(priority)
            batch.append(data)
            idxs.append(idx)

        # 중요도 샘플링(IS) 가중치 계산
        # 샘플링 확률 P(i) = priority / total_priority
        sampling_probabilities = priorities / self.tree.total()
        
        # IS 가중치 w_i = (N * P(i))^(-beta)
        # N = self.tree.number_of_entries (현재 버퍼에 저장된 데이터 수)
        # 우선순위가 높아서 자주 뽑힌 샘플(P(i)가 높음)은 가중치(w_i)를 낮게 주어,
        # 학습에 적은 영향을 미치도록 하여 편향(bias)을 보정
        importance_sampling_weight = np.power(self.tree.number_of_entries * sampling_probabilities, -self.beta)
        
        # 안정적인 학습을 위해 가중치를 최대 가중치로 정규화 (w_i = w_i / max(w_j))
        importance_sampling_weight /= importance_sampling_weight.max()

        # (샘플링된 배치, 샘플의 트리 인덱스, IS 가중치)를 반환
        return batch, idxs, importance_sampling_weight

    def update(self, idx, error):
        """
        학습이 끝난 후, 샘플링되었던 경험(idx)의 TD-Error(error)를 새로운 값으로 갱신
        """
        priority = self._get_priority(error)  # 새 TD-Error로 새 우선순위 계산
        self.tree.update(idx, priority)     # SumTree의 해당 인덱스 값을 갱신

    def __len__(self):
        """버퍼에 현재 저장된 경험의 수를 반환"""
        return self.tree.number_of_entries