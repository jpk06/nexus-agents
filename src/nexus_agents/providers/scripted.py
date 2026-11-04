"""Deterministic provider used to test agents without spending a token.

Testing an agent loop with a real LLM is slow, flaky and expensive. This
provider replays a script instead, so the loop's control flow — tool
dispatch, error recovery, step limits — can be asserted exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ..types import (
    FinishReason,
    Message,
    ToolCall,
    Usage,
    assistant_message,
)
from .base import GenerateRequest, LLMResponse

__all__ = ["ScriptedProvider", "tool_call_message"]

#: 脚本里允许出现的三种形态：完整消息、纯文本、或按请求动态生成
ScriptedResponse = Message | str | Callable[[GenerateRequest], "Message | str"]


def tool_call_message(
    name: str, arguments: dict[str, Any] | None = None, call_id: str = "call_1"
) -> Message:
    """Convenience builder for an assistant message that requests a tool."""
    return assistant_message(
        tool_calls=(ToolCall(id=call_id, name=name, arguments=arguments or {}),)
    )


class ScriptedProvider:
    """Replay a fixed list of responses, recording every request it saw."""

    def __init__(
        self,
        responses: Sequence[ScriptedResponse] = (),
        *,
        repeat_last: bool = True,
        usage: Usage | None = None,
        finish_reason: FinishReason = FinishReason.STOP,
    ) -> None:
        self._responses: list[ScriptedResponse] = list(responses)
        self._repeat_last = repeat_last
        self._usage = usage or Usage(prompt_tokens=1, completion_tokens=1)
        self._finish_reason = finish_reason
        self._cursor = 0
        self.requests: list[GenerateRequest] = []

    # ------------------------------------------------------------ 观测

    @property
    def call_count(self) -> int:
        return len(self.requests)

    @property
    def last_request(self) -> GenerateRequest | None:
        return self.requests[-1] if self.requests else None

    def reset(self) -> None:
        self._cursor = 0
        self.requests.clear()

    # ------------------------------------------------------------ 核心

    def generate(self, request: GenerateRequest) -> LLMResponse:
        self.requests.append(request)
        return self._replay(request)

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        """Async replay.

        本身没有 IO 可等待，但仍然实现 —— 否则 ``arun`` 会把整个测试替身丢进
        线程池，异步路径（并发、取消、异常传播）就永远测不到了。
        """
        self.requests.append(request)
        return self._replay(request)

    def _replay(self, request: GenerateRequest) -> LLMResponse:
        item = self._next_item()

        if callable(item):
            item = item(request)

        if isinstance(item, str):
            message = assistant_message(item)
        elif isinstance(item, Message):
            message = item
        else:  # pragma: no cover - 类型系统已约束，这里兜底
            message = assistant_message("")

        finish_reason = (
            FinishReason.TOOL_CALLS if message.has_tool_calls else self._finish_reason
        )
        return LLMResponse(message=message, usage=self._usage, finish_reason=finish_reason)

    def _next_item(self) -> ScriptedResponse | None:
        if self._cursor < len(self._responses):
            item = self._responses[self._cursor]
            self._cursor += 1
            return item
        if self._repeat_last and self._responses:
            return self._responses[-1]
        return None  # 脚本耗尽：返回空消息，让 agent 正常收尾
