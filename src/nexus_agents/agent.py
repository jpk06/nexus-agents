"""The ReAct agent loop.

控制流（就这几条，刻意保持简单）
--------------------------------
1. 把用户输入（可选 system prompt）交给 provider。
2. 如果模型要调工具 —— 逐个执行，把结果作为 ``role=tool`` 消息追加回对话。
3. 如果模型不再要工具 —— 它的文本就是最终答案，结束。
4. 跑满 ``max_steps`` 还没结束 —— 停下来并如实报告，而不是假装成功。

Why serial tool execution
-------------------------
Parallel tool calls look faster but make traces non-deterministic and error
handling much harder. Serial is the right default; a parallel strategy can
be added later behind the same interface.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .errors import AgentError, MaxStepsExceeded
from .memory import ContextPolicy, ConversationMemory
from .providers.base import GenerateRequest, LLMProvider, LLMResponse
from .tools import ToolRegistry, ToolSpec
from .tracing import EventType, TraceEvent, Tracer
from .types import Message, Usage, system_message, tool_result_message, user_message

__all__ = ["AgentConfig", "Stopped", "AgentResult", "ReActAgent"]


class Stopped(str, Enum):
    """Why an agent run ended."""

    #: 模型给出了不带工具调用的最终答案
    FINAL = "final"
    #: 用满了 max_steps
    MAX_STEPS = "max_steps"
    #: 工具连续出错，继续下去没有意义
    TOOL_ERROR = "tool_error"
    #: provider 返回了空消息
    EMPTY = "empty"


@dataclass(frozen=True)
class AgentConfig:
    """Tuning knobs for a run. Immutable so runs can't leak state."""

    max_steps: int = 12
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    #: 一次工具调用失败就立刻终止（默认否：把错误回喂，让模型自我修正）
    stop_on_tool_error: bool = False
    #: 连续失败到这个次数就放弃
    max_consecutive_errors: int = 3
    #: 每次调模型前按预算裁剪上下文；None 表示原样发送（旧行为）
    context_policy: ContextPolicy | None = None

    def __post_init__(self) -> None:
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")


@dataclass(frozen=True)
class AgentResult:
    """Everything a caller needs after a run, including the audit trail."""

    answer: str
    stopped_reason: Stopped
    steps: int
    messages: tuple[Message, ...]
    usage: Usage
    trace: tuple[TraceEvent, ...]

    @property
    def succeeded(self) -> bool:
        return self.stopped_reason is Stopped.FINAL

    @property
    def tool_calls(self) -> tuple[TraceEvent, ...]:
        return tuple(e for e in self.trace if e.type == EventType.TOOL_CALL)

    def raise_for_status(self) -> AgentResult:
        """Raise if the run did not finish cleanly. Returns self otherwise."""
        if self.succeeded:
            return self
        if self.stopped_reason is Stopped.MAX_STEPS:
            raise MaxStepsExceeded(self.steps)
        raise AgentError(f"Agent stopped: {self.stopped_reason.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "stopped_reason": self.stopped_reason.value,
            "steps": self.steps,
            "usage": {
                **self.usage.model_dump(),
                "total_tokens": self.usage.total_tokens,
            },
            "messages": [m.model_dump(mode="json") for m in self.messages],
            "trace": [e.to_dict() for e in self.trace],
        }


class ReActAgent:
    """A single agent: one provider, one tool set, one loop."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: Iterable[ToolSpec] | ToolRegistry = (),
        *,
        config: AgentConfig | None = None,
        name: str = "agent",
    ) -> None:
        self.provider = provider
        self.name = name
        self.config = config or AgentConfig()
        self.registry = (
            tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        )

    # ------------------------------------------------------------ 只读视图

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return self.registry.specs()

    def __repr__(self) -> str:
        return f"ReActAgent(name={self.name!r}, tools={list(self.registry.names)})"

    # ------------------------------------------------------------ 主循环

    def run(
        self,
        user_input: str,
        *,
        system: str | None = None,
        memory: ConversationMemory | None = None,
    ) -> AgentResult:
        """Run the loop to completion and return a full :class:`AgentResult`.

        这个方法是纯函数式的：不改变 agent 自身状态，同一输入 + 同一
        provider 脚本 => 同一结果，因此完全可测试。

        例外只有一处刻意为之：传入 ``memory`` 时，这一轮问答应在跑完后写回
        长期记忆 —— 记忆本来就是跨运行的副作用，藏起来反而更难推理。
        """
        config = self.config
        tracer = Tracer()
        messages: list[Message] = []

        prompt = system if system is not None else config.system_prompt
        recalled = memory.recall_messages(user_input) if memory is not None else ()
        if prompt:
            messages.append(system_message(prompt))
        messages.extend(recalled)
        messages.append(user_message(user_input))

        tracer.emit(
            EventType.AGENT_START,
            input=user_input,
            tools=list(self.registry.names),
            max_steps=config.max_steps,
            recalled=len(recalled),
        )
        if recalled:
            tracer.emit(EventType.MEMORY_RECALL, count=len(recalled))

        usage = Usage()
        step = 0
        answer = ""
        stopped = Stopped.EMPTY
        # 连续失败计数跨轮累计：模型反复用错工具时才放弃，
        # 而不是每一步重新开始数（那样永远触发不了上限）
        consecutive_errors = 0

        while step < config.max_steps:
            step += 1
            # 完整历史留在 messages 里（审计、结果回放都靠它），
            # 只有发给模型的那一份被裁剪
            request_messages = tuple(messages)
            if config.context_policy is not None:
                context = config.context_policy.apply(tuple(messages))
                request_messages = context.messages
                if context.stats.dropped or context.stats.had_summary:
                    tracer.emit(
                        EventType.CONTEXT_TRIMMED, step=step, **context.stats.to_dict()
                    )
            request = GenerateRequest(
                messages=request_messages,
                tools=self.registry.specs(),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            with tracer.span(EventType.LLM_CALL, step=step, message_count=len(messages)):
                response: LLMResponse = self.provider.generate(request)
            usage = usage + response.usage

            assistant_msg = response.message
            messages.append(assistant_msg)
            tracer.emit(
                EventType.LLM_RESPONSE,
                step=step,
                finish_reason=response.finish_reason.value,
                tool_calls=[c.name for c in assistant_msg.tool_calls],
                content_length=len(assistant_msg.content),
            )

            # 没有工具调用 => 模型认为自己说完了
            if not assistant_msg.tool_calls:
                answer = assistant_msg.content
                stopped = Stopped.FINAL if answer else Stopped.EMPTY
                break

            for call in assistant_msg.tool_calls:
                tracer.emit(
                    EventType.TOOL_CALL,
                    step=step,
                    tool=call.name,
                    arguments=call.arguments,
                )
                result = self.registry.invoke(call)
                tracer.emit(
                    EventType.TOOL_RESULT,
                    step=step,
                    tool=call.name,
                    is_error=result.is_error,
                    output=(result.error or result.output)[:500],
                )
                messages.append(tool_result_message(result))

                if not result.is_error:
                    consecutive_errors = 0  # 有一次成功就重新计数
                    continue

                consecutive_errors += 1
                answer = result.error or ""
                if (
                    config.stop_on_tool_error
                    or consecutive_errors >= config.max_consecutive_errors
                ):
                    stopped = Stopped.TOOL_ERROR
                    tracer.emit(
                        EventType.ERROR,
                        step=step,
                        tool=call.name,
                        error=answer,
                    )
                    break

            if stopped is Stopped.TOOL_ERROR:
                break
        else:
            # while 条件耗尽（不是 break）=> 真的跑满了步数
            stopped = Stopped.MAX_STEPS
            answer = _last_assistant_text(messages)
            tracer.emit(EventType.MAX_STEPS, step=step, max_steps=config.max_steps)

        tracer.emit(
            EventType.AGENT_END,
            step=step,
            stopped_reason=stopped.value,
            total_tokens=usage.total_tokens,
        )

        if memory is not None and answer:
            memory.remember_turn(user_input, answer, tags=(self.name,))

        return AgentResult(
            answer=answer,
            stopped_reason=stopped,
            steps=step,
            messages=tuple(messages),
            usage=usage,
            trace=tuple(tracer.events),
        )


def _last_assistant_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role.value == "assistant" and message.content:
            return message.content
    return ""
