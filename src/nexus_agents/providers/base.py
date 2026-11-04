"""Provider contracts: what an LLM backend must accept and return.

这一层是整个框架唯一依赖外部世界的边界。把它做成协议（Protocol）而不是
基类，是为了让测试替身、缓存装饰器、fallback 链都能无痛替换进来。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..tools import ToolSpec
from ..types import FinishReason, Message, Usage

__all__ = [
    "GenerateRequest",
    "LLMResponse",
    "LLMProvider",
    "AsyncLLMProvider",
    "supports_async",
]


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
    """Minimal interface every backend must satisfy.

    异步是**可选能力**，不是第二条必选契约：只实现 ``generate`` 的 provider
    仍然完全合法，``ReActAgent.arun`` 会把它放进线程池调用。这样第三方实现
    不必为了兼容异步而改一圈，异步收益却仍然拿得到 —— 只是拿不到「真异步」
    （并发请求、流式响应）那一部分。
    """

    def generate(self, request: GenerateRequest) -> LLMResponse: ...


@runtime_checkable
class AsyncLLMProvider(Protocol):
    """A backend that can be awaited directly.

    实现它的 provider 在 ``arun`` 下不会被丢进线程池，因此一次 agent 运行
    里可以有多个请求真正并发（异步工具、并行分支都靠这一点）。
    """

    async def agenerate(self, request: GenerateRequest) -> LLMResponse: ...


def supports_async(provider: object) -> bool:
    """Whether ``provider`` exposes a usable native async entry point."""
    return callable(getattr(provider, "agenerate", None))
