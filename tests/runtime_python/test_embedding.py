import torch
import unittest
import numpy as np
import runtime_kernel

class EmbeddingTest(unittest.TestCase):
    def test_embedding_op(self):
        # 1. Define tensor dimensions
        BATCH_SIZE = 3
        VOCAB_SIZE = 32 * 1024
        HIDDEN_SIZE = 4096
        
        # 2. Create torch tensors for the test
        input_data = torch.randint(0, VOCAB_SIZE, (1, BATCH_SIZE), dtype=torch.int64, device="cuda")
        weight_data = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda")
        output_data = torch.zeros(BATCH_SIZE, HIDDEN_SIZE, dtype=torch.float32, device="cuda")

        # 3. Launch the kernel directly via the pybind wrapper
        runtime_kernel.embedding(input_data, weight_data, output_data)
        
        # 4. Verification
        # The kernel looks up the embedding for each ID in the input tensor.
        # It's not just the first one. Let's verify each one.
        for i in range(BATCH_SIZE):
            selected_token_id = input_data[0, i].item()
            expected_embedding = weight_data[selected_token_id]
            self.assertTrue(torch.allclose(output_data[i], expected_embedding, atol=1e-6), 
                            f"Output row {i} does not match the expected embedding for token id {selected_token_id}.")
        
        print("\nEmbedding kernel unit test passed!")

if __name__ == "__main__":
    unittest.main() 