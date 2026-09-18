"""Generation settings, validated once independently of HTTP and ring transport."""
from __future__ import annotations

import math
import struct
from typing import Annotated, Mapping
from types import MappingProxyType

from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, field_serializer

from mirage import serving_config as abi

StopSequence = Annotated[tuple[TokenId, ...], Field(min_length=1, max_length=abi.MAX_STOP_TOKENS)]


class SamplingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, validate_default=True)

    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: StrictInt = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1, gt=0)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**63 - 1)
    logit_bias: dict[int, float] = Field(default_factory=dict, max_length=abi.MAX_BIASES)
    cache_history: bool = True
    stop_token_sequences: tuple[StopSequence, ...] = Field(default=(), max_length=abi.MAX_STOPS)
    
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
            raise ValueError("repetition_penalty must fit a positive finite float32") from exc
        if repetition == 0 or not math.isfinite(repetition):
            raise ValueError("repetition_penalty must fit a positive finite float32")
        return value


def validate_stops(stops):
    if len(stops) > abi.MAX_STOPS or any(not s for s in stops):
        raise ValueError(f"stop must contain one to {abi.MAX_STOPS} nonempty strings")
    return stops


class SamplingParams(SamplingOptions):
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: StrictInt = Field(default=0, ge=0, le=2**63 - 1)
    max_new_tokens: StrictInt | None = Field(default=None, gt=0)
    stop: tuple[str, ...] = ()

    _check_stops = field_validator("stop")(validate_stops)

    def pack(self, prompt_len, max_seq_length, vocab_size, eos_ids):
        """Bind validated options to model limits and serialize the ring payload."""
        remaining = max_seq_length - prompt_len
        budget = self.max_new_tokens if self.max_new_tokens is not None else remaining
        if prompt_len < 1 or budget < 1 or budget > remaining:
            raise ValueError("prompt plus requested output exceeds the context capacity")
        if len(eos_ids) > abi.MAX_EOS or any(not 0 <= t < vocab_size for t in eos_ids):
            raise ValueError("invalid model EOS token IDs")
        if any(not 0 <= t < vocab_size for t in self.logit_bias):
            raise ValueError("logit_bias token ID exceeds model vocabulary")
        if any(t >= vocab_size for seq in self.stop_token_sequences for t in seq):
            raise ValueError("stop token ID exceeds model vocabulary")
        words = [0] * abi.CONFIG_WORDS
        for name, value in (
            ("MAX_NEW_TOKENS", budget), ("SEED", self.seed), ("VOCAB_SIZE", vocab_size),
            ("TOP_K", min(self.top_k, vocab_size)), ("EOS_COUNT", len(eos_ids)),
            ("BIAS_COUNT", len(self.logit_bias)), ("CACHE_HISTORY", int(self.cache_history)),
            ("STOP_COUNT", len(self.stop_token_sequences)),
        ):
            words[getattr(abi, name)] = value
        for name in ("temperature", "top_p", "frequency_penalty", "presence_penalty", "repetition_penalty"):
            words[getattr(abi, name.upper())] = _bits(getattr(self, name))
        words[abi.EOS_IDS:abi.EOS_IDS + len(eos_ids)] = eos_ids
        for i, (token, bias) in enumerate(sorted(self.logit_bias.items())):
            start = abi.BIASES + abi.BIAS_STRIDE * i
            words[start:start + abi.BIAS_STRIDE] = [token, _bits(bias)]
        for i, sequence in enumerate(self.stop_token_sequences):
            start = abi.STOP_SEQUENCES + abi.STOP_STRIDE * i
            words[start:start + len(sequence) + 1] = [len(sequence), *sequence]
        return words


def _bits(value):
    return struct.unpack("q", struct.pack("d", value))[0]
