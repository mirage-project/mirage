import torch
import unittest
import numpy as np
import runtime_kernel

class ArgmaxTest(unittest.TestCase):
    def test_argmax_op(self):

        BATCH_NUM = 3 # The C++ wrapper assumes BATCH_SIZE = 1
        BATCH_SIZE = 2 # The C++ wrapper assumes BATCH_SIZE = 1
        VOCAB_SIZE = 153600 # Must be divisible by NUM_PARTIAL_TASKS
        NUM_PARTIAL_TASKS = 96
        n_row = BATCH_NUM * BATCH_SIZE
        
        # Using bfloat16 as this is what the kernel expects from the wrapper
        input_data = torch.randn(n_row, VOCAB_SIZE, dtype=torch.bfloat16, device="cuda")
        
        # Place a max value at a known, random index
        max_val = torch.tensor([65504.0], dtype=torch.bfloat16, device="cuda") # Max representable bfloat16
        # max_idx = np.random.randint(0, VOCAB_SIZE)
        max_idx_per_row = np.random.randint(0, VOCAB_SIZE, n_row)
        for i in range(n_row):
            input_data[i, max_idx_per_row[i]] = max_val

        # The kernel will write the final index to this tensor
        output_result = torch.empty([1, n_row], dtype=torch.int64, device="cuda")
        medium_output_result_idx = torch.empty([n_row, 96 // (BATCH_NUM)], dtype=torch.int64, device="cuda").contiguous()
        medium_output_result_val = torch.empty([n_row, 96 // (BATCH_NUM)], dtype=torch.bfloat16, device="cuda").contiguous()
        
        runtime_kernel.argmax(input_data, output_result, medium_output_result_idx, medium_output_result_val)
        
        # Use numpy's testing utility to compare arrays element-wise.
        # Squeeze the output to 1D to match the shape of the expected indices.
        np.testing.assert_array_equal(output_result.cpu().numpy().squeeze(), max_idx_per_row)
        
        print("\nArgmax kernel unit test passed!")


if __name__ == "__main__":
    unittest.main()
