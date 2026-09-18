"""Thread-safe tokenizer/detokenizer wrapper."""

from __future__ import annotations

import copy
import json
import threading


class TokenizerManager:
    """Wraps a HuggingFace tokenizer for thread-safe tokenization/detokenization."""

    def __init__(self, tokenizer, developer_role="reject") -> None:
        self._tokenizer = tokenizer
        self._lock = threading.Lock()
        self.developer_role = developer_role

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
        messages = copy.deepcopy(messages)
        if any(m["role"] == "tool" or m.get("tool_calls") for m in messages):
            template = self._tokenizer.chat_template
            if not isinstance(template, str) or "tool_calls" not in template:
                raise ValueError("model chat template does not support tool-call history")
        for message in messages:
            if message["role"] == "developer" and self.developer_role != "native":
                if self.developer_role == "reject":
                    raise ValueError("model requires an explicit developer-role adapter")
                message["role"] = "system"
            # HF templates expect parsed function arguments; API history uses JSON strings.
            for call in message.get("tool_calls", []):
                try:
                    call["function"]["arguments"] = json.loads(call["function"]["arguments"])
                except (ValueError, TypeError) as exc:
                    raise ValueError("tool call arguments must be valid JSON") from exc
        with self._lock:
            if not self._tokenizer.chat_template:
                raise ValueError("model tokenizer has no chat template")
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
