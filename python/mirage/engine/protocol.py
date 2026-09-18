"""Text OpenAI request schemas with shared sampling validation."""
from __future__ import annotations

import secrets
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .sampling import SamplingOptions, SamplingParams, validate_stops
# Compatibility exports; the definition is shared with the CUDA runtime.
from mirage.serving_config import CONFIG_WORDS, MAX_BIASES, MAX_EOS


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class TextPart(APIModel):
    type: Literal["text"]
    text: str


class FunctionCall(APIModel):
    name: str
    arguments: str


class ToolCall(APIModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class Message(APIModel):
    role: Literal["system", "developer", "user", "assistant", "tool"]
    content: str | list[TextPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None

    @model_validator(mode="after")
    def check_role(self):
        if self.content is None and not (self.role == "assistant" and self.tool_calls):
            raise ValueError("content is required except for assistant tool calls")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required only for tool messages")
        if self.tool_calls is not None and self.role != "assistant":
            raise ValueError("tool_calls requires the assistant role")
        return self

    def template_message(self):
        value = self.model_dump(exclude_none=True)
        if isinstance(self.content, list):
            value["content"] = "".join(part.text for part in self.content)
        return value


class StreamOptions(APIModel):
    include_usage: bool = False


class CompletionRequest(SamplingOptions):
    model: str
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: StrictInt | None = Field(default=None, gt=0)
    max_completion_tokens: StrictInt | None = Field(default=None, gt=0)
    stop: str | list[str] | None = None
    n: Literal[1] = 1
    user: str | None = None

    @model_validator(mode="after")
    def check_options(self):
        if self.max_tokens is not None and self.max_completion_tokens is not None:
            if self.max_tokens != self.max_completion_tokens:
                raise ValueError("max_tokens and max_completion_tokens conflict")
        validate_stops([self.stop] if isinstance(self.stop, str) else self.stop or [])
        if self.stream_options is not None and not self.stream:
            raise ValueError("stream_options requires stream=true")
        return self

    def sampling_params(self):
        # The shared sampling fields have already been validated at ingress.
        options = {name: getattr(self, name) for name in SamplingOptions.model_fields}
        options["seed"] = self.seed if self.seed is not None else secrets.randbits(63)
        return SamplingParams.model_construct(
            **options, max_new_tokens=self.max_completion_tokens or self.max_tokens,
            stop=tuple([self.stop] if isinstance(self.stop, str) else self.stop or []),
        )


class ChatRequest(CompletionRequest):
    messages: list[Message] = Field(min_length=1)

    @model_validator(mode="after")
    def check_tool_history(self):
        pending = set()
        seen = set()
        for message in self.messages:
            if pending and message.role != "tool":
                raise ValueError("assistant tool calls must be followed by their tool results")
            if message.role == "tool":
                if message.tool_call_id not in pending:
                    raise ValueError("tool result has no matching assistant tool call")
                pending.remove(message.tool_call_id)
            for call in message.tool_calls or []:
                if call.id in seen:
                    raise ValueError("duplicate tool call ID")
                pending.add(call.id)
                seen.add(call.id)
        if pending:
            raise ValueError("missing tool results")
        return self


class TextRequest(CompletionRequest):
    prompt: str
