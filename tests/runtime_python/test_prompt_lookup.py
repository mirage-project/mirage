import torch
import unittest
import numpy as np
import runtime_kernel

class PromptLookupTest(unittest.TestCase):
    def test_prompt_lookup_op(self):
        # 1. Define test parameters
        NGRAM_SIZE = 3
        SPEC_LENGTH = 5
        VOCAB_SIZE = 1000
        SEARCH_LEN = 512
        PROMPT_LEN = SEARCH_LEN + NGRAM_SIZE
        
        # 2. Create test data
        all_tokens = torch.randint(40, VOCAB_SIZE, (1, 2048), dtype=torch.int64, device="cuda")
        
        # Plant the n-gram to be found
        ngram = torch.tensor([10, 20, 30], dtype=torch.int64, device="cuda")
        first_occurrence_idx = 100
        all_tokens[0, first_occurrence_idx : first_occurrence_idx + NGRAM_SIZE] = ngram
        
        # Plant the speculative tokens that should be returned upon match
        speculative_tokens = torch.tensor([4, 5, 6, 7, 8], dtype=torch.int64, device="cuda")
        all_tokens[0, first_occurrence_idx + NGRAM_SIZE : first_occurrence_idx + NGRAM_SIZE + SPEC_LENGTH] = speculative_tokens
        
        # Plant the n-gram at the end of the prompt to define what we are searching for
        all_tokens[0, PROMPT_LEN - NGRAM_SIZE : PROMPT_LEN] = ngram
        
        # Define the normally decoded token (at position prompt_len)
        next_regular_token = torch.tensor([99], dtype=torch.int64, device="cuda")
        all_tokens[0, PROMPT_LEN] = next_regular_token

        # 3. Call the kernel
        # Output now has size SPEC_LENGTH + 1
        output_result = torch.empty(SPEC_LENGTH + 1, dtype=torch.int64, device="cuda")
        runtime_kernel.prompt_lookup(all_tokens, PROMPT_LEN, NGRAM_SIZE, SPEC_LENGTH, output_result)

        # 4. Verification for match case
        expected_result = torch.cat([next_regular_token, speculative_tokens])
        self.assertTrue(torch.equal(output_result, expected_result),
                        f"Output {output_result} does not match expected {expected_result}")
        print("\nPrompt lookup kernel test passed!")

        # 5. Test case with no match
        all_tokens_no_match = torch.randint(40, VOCAB_SIZE, (1, 2048), dtype=torch.int64, device="cuda")
        all_tokens_no_match[0, PROMPT_LEN - NGRAM_SIZE : PROMPT_LEN] = ngram
        all_tokens_no_match[0, PROMPT_LEN] = next_regular_token # Still need the regular token

        output_no_match = torch.empty(SPEC_LENGTH + 1, dtype=torch.int64, device="cuda")
        runtime_kernel.prompt_lookup(all_tokens_no_match, PROMPT_LEN, NGRAM_SIZE, SPEC_LENGTH, output_no_match)
        
        expected_no_match = torch.full((SPEC_LENGTH + 1,), -1, dtype=torch.int64, device="cuda")
        expected_no_match[0] = next_regular_token
        self.assertTrue(torch.equal(output_no_match, expected_no_match),
                        f"Output {output_no_match} does not match expected {expected_no_match} for no-match case.")
        print("Prompt lookup kernel no-match case passed!")


if __name__ == "__main__":
    unittest.main() 