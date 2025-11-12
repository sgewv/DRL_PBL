import unittest
import torch

from src.replay_buffer import ReplayBuffer, Transition

class TestReplayBuffer(unittest.TestCase):
    """
    ### 표준 리플레이 버퍼(`ReplayBuffer`) 기능 검증 테스트
    
    #### 설계 의도 (Design Intent)
    - 단위 테스트(Unit Test): `ReplayBuffer`라는 단일 컴포넌트가 의도된 대로 정확하게 동작하는지
      독립적으로 검증.
    - 핵심 기능 검증: 버퍼의 가장 기본적인 두 가지 기능인 '데이터 저장(push)'과 '무작위 샘플링(sample)',
      그리고 '용량 제한(capacity)' 기능이 올바르게 작동하는지 확인.
    """

    def test_push_and_sample(self):
        """
        #### 테스트 목표
        - 버퍼에 데이터가 정상적으로 `push`(저장)되는지 확인.
        - 저장된 데이터로부터 요청한 `batch_size`만큼 정확하게 `sample`(샘플링)되는지 확인.
        
        #### 테스트 절차
        1. 용량이 100인 버퍼를 생성.
        2. 10개의 더미(dummy) 트랜지션을 버퍼에 `push`.
        3. 버퍼의 현재 크기(`len(buffer)`)가 10인지 `assertEqual`로 확인.
        4. 버퍼에서 5개의 배치를 `sample`.
        5. 샘플링된 배치의 크기가 5인지 `assertEqual`로 확인.
        6. 샘플링된 데이터의 타입이 `Transition` 객체인지 `assertIsInstance`로 확인.
        """
        buffer = ReplayBuffer(100)
        
        # 테스트용 더미 데이터 생성
        dummy_transition = Transition(
            state=torch.randn(4),
            action=torch.tensor([1]),
            reward=torch.tensor([1.0]),
            next_state=torch.randn(4),
            done=torch.tensor([False])
        )
        
        # 10개의 트랜지션 저장
        for _ in range(10):
            buffer.push(*dummy_transition)
            
        self.assertEqual(len(buffer), 10, "10개의 트랜지션 push 후 버퍼 크기가 10이어야 함")
        
        # 5개 배치 샘플링
        sample = buffer.sample(5)
        self.assertEqual(len(sample), 5, "5개 배치 샘플링 후 샘플 크기가 5여야 함")
        self.assertIsInstance(sample[0], Transition, "샘플링된 아이템은 Transition 객체여야 함")

    def test_capacity(self):
        """
        #### 테스트 목표
        - 버퍼의 용량(`capacity`)이 가득 찼을 때, 가장 오래된 데이터가 순서대로 삭제되는지(FIFO) 확인.
        
        #### 테스트 절차
        1. 용량이 10인 버퍼를 생성.
        2. 15개의 트랜지션을 버퍼에 `push`. (0부터 14까지의 값을 가진 상태)
        3. 버퍼의 최종 크기가 설정된 용량인 10과 같은지 `assertEqual`로 확인.
        4. 버퍼의 모든 데이터를 샘플링하여, 가장 먼저 들어간 데이터(0~4)는 삭제되고,
           나중에 들어간 데이터(5~14)만 남아있는지 `assertNotIn`과 `assertIn`으로 확인.
        """
        capacity = 10
        buffer = ReplayBuffer(capacity)
        
        # 용량을 초과하여 15개의 트랜지션 저장
        for i in range(15):
            buffer.push(torch.tensor([float(i)]), None, None, None, None)
            
        self.assertEqual(len(buffer), capacity, f"버퍼 용량({capacity}) 초과 시 크기는 용량과 같아야 함")
        
        # FIFO 원칙에 따라 오래된 데이터가 삭제되었는지 확인
        sample = buffer.sample(capacity)
        states = [item.state.item() for item in sample]
        
        # 0부터 4까지의 초기 데이터는 삭제되었어야 함
        self.assertNotIn(0.0, states, "가장 오래된 데이터(0.0)는 버퍼에서 삭제되었어야 함")
        self.assertNotIn(4.0, states, "5번째로 오래된 데이터(4.0)는 버퍼에서 삭제되었어야 함")
        
        # 5부터 14까지의 최신 데이터는 남아있어야 함
        self.assertIn(5.0, states, "가장 오래된 데이터가 삭제된 후 남은 첫 데이터(5.0)가 있어야 함.")
        self.assertIn(14.0, states, "가장 최신 데이터(14.0)가 버퍼에 있어야 함.")

if __name__ == '__main__':
    unittest.main()
