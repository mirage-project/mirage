"""Text OpenAI request schemas with shared sampling validation."""
from __future__ import annotations

import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .sampling import SamplingParams
from mirage import core


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextPart(APIModel):
    type: Literal["text"]
    text: str


class Message(APIModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[TextPart]

    def template_message(self):
        value = self.model_dump(exclude_none=True)
        if isinstance(self.content, list):
            value["content"] = "".join(part.text for part in self.content)
        return value


class StreamOptions(APIModel):
    include_usage: bool = False


class CompletionRequest(APIModel):
    model: str

    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=1.0, gt=0, le=1)
    top_k: StrictInt = Field(default=0, ge=0)
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    repetition_penalty: float = Field(default=1, gt=0)
    seed: StrictInt | None = Field(default=None, ge=0, le=2**63 - 1)
    logit_bias: dict[int, float] = Field(
        default_factory=dict,
        max_length=core.serving_max_biases(),
    )

    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: StrictInt | None = Field(default=None, gt=0)
    max_completion_tokens: StrictInt | None = Field(default=None, gt=0)
    n: Literal[1] = 1
    user: str | None = None

    @model_validator(mode="after")
    def check_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            if self.max_tokens != self.max_completion_tokens:
                raise ValueError("max_tokens and max_completion_tokens conflict")
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        return self

    def sampling_params(self):
        return SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed if self.seed is not None else secrets.randbits(63),
            logit_bias=self.logit_bias,
            max_new_tokens=self.max_completion_tokens or self.max_tokens,
        )


class ChatRequest(CompletionRequest):
    messages: list[Message] = Field(min_length=1)


class TextRequest(CompletionRequest):
    prompt: str
