"""Thread-safe tokenizer/detokenizer wrapper."""

from __future__ import annotations

import threading


class TokenizerManager:
    """Wraps a HuggingFace tokenizer for thread-safe tokenization/detokenization."""

    def __init__(self, tokenizer) -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()

    def tokenize(self, prompt: str, use_template: bool = True) -> list[int]:
        """Apply chat template (if requested) and return token IDs."""
        if use_template:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ]
            text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
        with self._lock:
            return self._tokenizer([text], return_tensors="pt").input_ids[0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text, skipping special tokens."""
        with self._lock:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True)

    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        with self._lock:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)
