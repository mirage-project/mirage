import torch
import unittest
import numpy as np
import runtime_kernel

class VerifyTest(unittest.TestCase):
    def test_verify_op_all_match(self):
        NUM_SPEC_TOKENS = 5
        
        spec_tokens = torch.tensor([100, 10, 20, 30, 40, 50], dtype=torch.int64, device="cuda")
        target_tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64, device="cuda")

        accepted_len = torch.empty((1,), dtype=torch.int32, device="cuda")
        new_tokens = torch.full((NUM_SPEC_TOKENS + 1,), -1, dtype=torch.int64, device="cuda")
        
        runtime_kernel.verify(spec_tokens, target_tokens, accepted_len, new_tokens)

        # Verification
        expected_len = NUM_SPEC_TOKENS + 1
        self.assertEqual(accepted_len.item(), expected_len)
        
        expected_new_tokens = np.array([10, 20, 30, 40, 50, 60])
        np.testing.assert_array_equal(new_tokens.cpu().numpy(), expected_new_tokens)
        print("\nVerify kernel 'all match' test passed!")

    def test_verify_op_partial_match(self):
        NUM_SPEC_TOKENS = 5
        
        # Input tensors
        # Mismatch at index 2 (spec is 99, target is 30)
        spec_tokens = torch.tensor([100, 10, 20, 99, 40, 50], dtype=torch.int64, device="cuda")
        target_tokens = torch.tensor([10, 20, 30, 40, 50, 60], dtype=torch.int64, device="cuda")

        # Output tensors
        accepted_len = torch.empty((1,), dtype=torch.int32, device="cuda")
        new_tokens = torch.full((NUM_SPEC_TOKENS + 1,), -1, dtype=torch.int64, device="cuda")
        
        runtime_kernel.verify(spec_tokens, target_tokens, accepted_len, new_tokens)

        # Verification
        # First 2 speculative tokens are accepted. Plus the new one from target_tokens. Total length is 3.
        expected_len = 3
        self.assertEqual(accepted_len.item(), expected_len)
        
        # The new sequence should be the first 2 correct ones, plus the new one.
        expected_new_tokens = np.array([10, 20, 30, -1, -1, -1])
        np.testing.assert_array_equal(new_tokens.cpu().numpy(), expected_new_tokens)
        print("\nVerify kernel 'partial match' test passed!")

if __name__ == "__main__":
    unittest.main() 