"""Generation settings, validated once independently of HTTP and ring transport."""
from __future__ import annotations

import math
import struct

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator

from mirage import core


class SamplingParams(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, validate_default=True)
    
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: StrictInt = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1, gt=0)
    seed: StrictInt = Field(default=42, ge=0, le=2**63 - 1)
    logit_bias: dict[int, float] = Field(default_factory=dict, max_length=core.serving_max_biases())
    cache_history: bool = True
    
    max_new_tokens: StrictInt | None = Field(default=None, gt=0)

    @field_validator("logit_bias")
    @classmethod
    def check_biases(cls, value):
        if any(not -100 <= bias <= 100 for bias in value.values()):
            raise ValueError("logit_bias values must be in [-100, 100]")
        return value


    @field_validator("repetition_penalty")
    @classmethod
    def check_repetition(cls, value):
        try:
            repetition = struct.unpack("f", struct.pack("f", value))[0]
        except OverflowError as exc:
            raise ValueError(
                "repetition_penalty must fit a positive finite float32"
            ) from exc

        if repetition == 0 or not math.isfinite(repetition):
            raise ValueError(
                "repetition_penalty must fit a positive finite float32"
            )
        return value

    def pack(self, prompt_len, max_seq_length, vocab_size, eos_ids):
        """Bind validated options to model limits and serialize the ring payload."""
        remaining = max_seq_length - prompt_len
        budget = self.max_new_tokens if self.max_new_tokens is not None else remaining
        
        if prompt_len < 1 or budget < 1 or budget > remaining:
            raise ValueError("prompt plus requested output exceeds the context capacity")
        
        if len(eos_ids) > core.serving_max_eos() or any(not 0 <= t < vocab_size for t in eos_ids):
            raise ValueError("invalid model EOS token IDs")
        
        if any(not 0 <= t < vocab_size for t in self.logit_bias):
            raise ValueError("logit_bias token ID exceeds model vocabulary")
        
        return core.pack_serving_config(
            budget, self.seed, vocab_size, self.temperature, self.top_p,
            min(self.top_k, vocab_size), self.frequency_penalty,
            self.presence_penalty, self.repetition_penalty,
            self.cache_history, eos_ids, self.logit_bias,
        )
