"""Incremental text decoding, stop matching, and generation events."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationEvent:
    text: str = ""
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def usage(self):
        return dict(prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                    total_tokens=self.prompt_tokens + self.completion_tokens)


class OutputProcessor:
    def __init__(self, tokenizer, stops=(), eos_ids=(), stop_token_sequences=()):
        self.tokenizer = tokenizer
        self.stops = stops
        self.eos_ids = set(eos_ids)
        self.stop_token_sequences = stop_token_sequences
        self.token_ids = []
        self.emitted = ""
        self.stopped = False

    def push(self, token):
        if self.stopped:
            return ""
        self.token_ids.append(token)
        if token in self.eos_ids:
            self.stopped = True
            return self.flush()
        token_hold = 0
        matched_tokens = 0
        for sequence in self.stop_token_sequences:
            for size in range(1, min(len(sequence), len(self.token_ids)) + 1):
                if tuple(self.token_ids[-size:]) == sequence[:size]:
                    token_hold = max(token_hold, size)
                    if size == len(sequence):
                        matched_tokens = max(matched_tokens, size)
        if matched_tokens:
            self.stopped = True
            token_hold = matched_tokens
        visible = self.token_ids[:-token_hold] if token_hold else self.token_ids
        text = self.tokenizer.decode(visible)
        matches = [text.find(stop) for stop in self.stops if stop in text]
        if matches:
            text = text[:min(matches)]
            self.stopped = True
        elif not self.stopped:
            # Incomplete byte sequences are not stable until later tokens arrive.
            text = text.rstrip("\ufffd")
            hold = 0
            for stop in self.stops:
                for size in range(1, min(len(stop), len(text) + 1)):
                    if text.endswith(stop[:size]):
                        hold = max(hold, size)
            if hold:
                text = text[:-hold]
        return self._delta(text)

    def _delta(self, text):
        if not text.startswith(self.emitted):
            raise RuntimeError("tokenizer changed already emitted text")
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    def flush(self):
        if self.stopped and self.token_ids and self.token_ids[-1] not in self.eos_ids:
            return ""
        return self._delta(self.tokenizer.decode(self.token_ids))
