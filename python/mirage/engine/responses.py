"""OpenAI response formatting, independent of scheduling and request lifetime."""
from dataclasses import dataclass, field
import json
import time
import uuid

from .output import GenerationEvent


def encode_sse(value):
    return "data: " + json.dumps(value, ensure_ascii=False) + "\n\n"


@dataclass(frozen=True)
class CompletionResponse:
    model: str
    chat: bool
    created: int = field(default_factory=lambda: int(time.time()))
    suffix: str = field(default_factory=lambda: uuid.uuid4().hex)

    def envelope(self, choices, *, stream=False, usage=None):
        result = dict(
            id=("chatcmpl-" if self.chat else "cmpl-") + self.suffix,
            created=self.created, model=self.model, choices=choices,
            object=("chat.completion.chunk" if stream else "chat.completion")
                   if self.chat else "text_completion",
        )
        if usage is not None:
            result["usage"] = usage
        return result

    def choice(self, text, reason=None, *, stream=False, role=False):
        result = dict(index=0, finish_reason=reason)
        if self.chat:
            content = {} if stream and reason else {"content": text}
            if role or not stream:
                content["role"] = "assistant"
            result["delta" if stream else "message"] = content
        else:
            result["text"] = text
        return result

    def initial_chunk(self):
        return self.envelope([self.choice("", stream=True, role=True)], stream=True)

    def chunks(self, event: GenerationEvent, include_usage=False):
        if event.text:
            yield self.envelope([self.choice(event.text, stream=True)], stream=True)
        if event.finish_reason is not None:
            yield self.envelope([self.choice("", event.finish_reason, stream=True)], stream=True)
            if include_usage:
                yield self.envelope([], stream=True, usage=event.usage)

    def completed(self, text, event: GenerationEvent):
        return self.envelope([self.choice(text, event.finish_reason)], usage=event.usage)
