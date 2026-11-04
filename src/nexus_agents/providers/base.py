"""Provider contracts: what an LLM backend must accept and return.

这一层是整个框架唯一依赖外部世界的边界。把它做成协议（Protocol）而不是
基类，是为了让测试替身、缓存装饰器、fallback 链都能无痛替换进来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..tools import ToolSpec
from ..types import FinishReason, Message, Usage

__all__ = ["GenerateRequest", "LLMResponse", "LLMProvider"]


@dataclass(frozen=True)
class GenerateRequest:
    """Everything a provider needs to produce one assistant message."""

    messages: tuple[Message, ...]
    tools: tuple[ToolSpec, ...] = ()
    temperature: float | None = None
    max_tokens: int | None = None
    #: 透传给 provider 的额外字段（如 response_format、top_p）
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("GenerateRequest requires at least one message")


@dataclass(frozen=True)
class LLMResponse:
    """A normalised provider reply, backend-agnostic."""

    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    #: 原始响应体，排障时很有用；不参与相等性判断以外的逻辑
    raw: dict[str, Any] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface every backend must satisfy."""

    def generate(self, request: GenerateRequest) -> LLMResponse: ...
