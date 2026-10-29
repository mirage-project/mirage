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

    def tokenize_messages(self, messages: list[dict]) -> list[int]:
        with self._lock:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True)

    def tokenize_raw(self, prompt: str) -> list[int]:
        """Tokenize an API text completion without adding a chat template."""
        with self._lock:
            ids = self._tokenizer.encode(prompt, add_special_tokens=False)
            if not ids:
                bos = self._tokenizer.bos_token_id
                if bos is None:
                    raise ValueError("empty prompt is unsupported by this tokenizer")
                ids = [bos]
            return ids

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text, skipping special tokens."""
        with self._lock:
            return self._tokenizer.decode(token_ids, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)

    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to text."""
        with self._lock:
            return self._tokenizer.decode([token_id], skip_special_tokens=True)
