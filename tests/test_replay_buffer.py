import unittest
import torch

from src.replay_buffer import ReplayBuffer, Transition

class TestReplayBuffer(unittest.TestCase):

    def test_push_and_sample(self):
        """Test if the buffer correctly stores and samples transitions."""
        buffer = ReplayBuffer(100)
        
        # Dummy data
        state = torch.randn(4)
        action = torch.tensor([1])
        reward = torch.tensor([1.0])
        next_state = torch.randn(4)
        done = False
        
        # Push 10 transitions
        for _ in range(10):
            buffer.push(state, action, reward, next_state, done)
            
        self.assertEqual(len(buffer), 10)
        
        # Sample a batch of 5
        sample = buffer.sample(5)
        self.assertEqual(len(sample), 5)
        
        # Check the type of the sampled item
        self.assertIsInstance(sample[0], Transition)

    def test_capacity(self):
        """Test if the buffer respects its capacity limit."""
        capacity = 10
        buffer = ReplayBuffer(capacity)
        
        # Push 15 transitions
        for i in range(15):
            state = torch.tensor([float(i)])
            buffer.push(state, None, None, None, None)
            
        self.assertEqual(len(buffer), capacity)
        
        # Check if the oldest items are discarded
        sample = buffer.sample(capacity)
        states = [item.state.item() for item in sample]
        self.assertNotIn(0.0, states)
        self.assertNotIn(4.0, states)
        self.assertIn(5.0, states)
        self.assertIn(14.0, states)

if __name__ == '__main__':
    unittest.main()
