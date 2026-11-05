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

Why one async core, not two loops
---------------------------------
``arun`` holds the real loop; ``run`` is a thin driver that executes it with
``asyncio.run``. The alternative — a sync copy and an async copy — drifts
within a week: someone fixes a stop condition in one and forgets the other,
and the two paths silently disagree about what the agent does.

The cost of this choice is explicit: ``run`` cannot be called from a thread
that already has a running event loop (that is what ``arun`` is for).

Why stream events, not tokens
-----------------------------
``astream`` yields what the loop actually knows: step boundaries, tool calls,
tool results, the final answer. It does **not** fake token streaming by
slicing the finished answer into characters — that looks impressive in a demo
and is worthless in production, because the user still waits for the whole
response before the first "token" appears.

Real token streaming needs a provider contract that emits deltas
(``AsyncIterator[str]`` plus incremental tool-call assembly). That is a
separate, provider-specific job; when it lands it plugs into the same loop and
adds one more event type. Until then, an SSE endpoint built on ``astream``
gives live progress — "calling search", "search returned 12 rows" — with no
lying about latency.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncGenerator, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, TypeAlias

from .errors import AgentError, MaxStepsExceeded
from .memory import ContextPolicy, ConversationMemory
from .providers.base import GenerateRequest, LLMProvider, LLMResponse
from .tools import ToolRegistry, ToolSpec
from .tracing import EventType, TraceEvent, Tracer
from .types import Message, Usage, system_message, tool_result_message, user_message

__all__ = ["AgentConfig", "Stopped", "AgentResult", "AgentStreamItem", "ReActAgent"]


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


#: 流里的一项：运行期间发生的 ``TraceEvent``，**最后**一项是 ``AgentResult``。
#: 复用事件类型而不是另造一套「流式事件」，是因为两者描述的是同一件事 ——
#: 多一套类型只会多一份要同步维护的映射。
AgentStreamItem: TypeAlias = TraceEvent | AgentResult


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
        history: Sequence[Message] | None = None,
    ) -> AgentResult:
        """Run the loop to completion and return a full :class:`AgentResult`.

        这个方法是纯函数式的：不改变 agent 自身状态，同一输入 + 同一
        provider 脚本 => 同一结果，因此完全可测试。

        例外只有一处刻意为之：传入 ``memory`` 时，这一轮问答应在跑完后写回
        长期记忆 —— 记忆本来就是跨运行的副作用，藏起来反而更难推理。

        实现上只是 ``asyncio.run(arun(...))``：真正跑循环的是 :meth:`arun`。
        在已有事件循环的线程里调用会抛 ``RuntimeError`` —— 这种情况应当
        ``await agent.arun(...)``，而不是让同步版本去猜怎么抢 loop。
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.arun(user_input, system=system, memory=memory, history=history)
            )
        raise RuntimeError(
            "run() cannot be called from a running event loop; "
            "use `await agent.arun(...)` instead."
        )

    async def arun(
        self,
        user_input: str,
        *,
        system: str | None = None,
        memory: ConversationMemory | None = None,
        history: Sequence[Message] | None = None,
    ) -> AgentResult:
        """Async version of :meth:`run`. Same loop, same result type.

        与同步版本唯一的语义差别在 IO 的等待方式：provider 有原生 ``agenerate``
        就直接 await，没有则放进线程池；工具走 ``ToolRegistry.ainvoke``，
        async 工具并行等待、同步工具同样进线程池。控制流本身一行都没有分叉。

        实现上是「把 :meth:`astream` 抽干，取最后一项」—— 这样「跑完」与
        「边跑边看」用的是同一段代码，不存在两份可能分叉的循环。
        """
        result: AgentResult | None = None
        async for item in self._run_stream(
            user_input, system=system, memory=memory, history=history
        ):
            if isinstance(item, AgentResult):
                result = item
        return _require_result(result)

    async def astream(
        self,
        user_input: str,
        *,
        system: str | None = None,
        memory: ConversationMemory | None = None,
        history: Sequence[Message] | None = None,
    ) -> AsyncGenerator[AgentStreamItem, None]:
        """Run the loop, yielding each event as it happens.

        消费方式（最后一项一定是 ``AgentResult``）::

            final = None
            async for item in agent.astream("查一下库存"):
                if isinstance(item, AgentResult):
                    final = item
                else:
                    print(item.type, item.payload)

        三条契约，都是能踩坑的地方：

        1. **最后一项必定是结果**。事件顺序与 ``result.trace`` 完全一致，
           所以「边看边流」和「跑完再看」不会给出两套事实。
        2. **中途放弃 = 取消**。``break`` 之后生成器被关闭，这一轮的记忆
           写回不会发生 —— 半途而废的问答不该被当成记忆。
        3. **背压是真的**：事件直接产出，不经过无界队列。消费得慢，agent
           就跟着慢。这是刻意的，否则一个卡住的 SSE 客户端会把内存吃满。

        注意 ``llm.call`` 事件在调用**发起时**就发出，因此它被消费时
        ``duration_ms`` 还是 ``None``（要靠后面的 ``llm.response`` 才知道
        花了多久）。这正是进度条需要的信息：先告诉你在等，再告诉你等了多久。
        """
        async for item in self._run_stream(
            user_input, system=system, memory=memory, history=history
        ):
            yield item

    async def _run_stream(
        self,
        user_input: str,
        *,
        system: str | None,
        memory: ConversationMemory | None,
        history: Sequence[Message] | None = None,
    ) -> AsyncGenerator[AgentStreamItem, None]:
        """The one and only control flow. ``arun`` drains it, ``astream`` relays it."""
        config = self.config
        tracer = Tracer()
        messages: list[Message] = []

        prompt = system if system is not None else config.system_prompt
        recalled = memory.recall_messages(user_input) if memory is not None else ()
        if prompt:
            messages.append(system_message(prompt))
        messages.extend(recalled)
        # history 是「逐字照搬的最近若干轮」，和 memory 的「相关性召回」是两种
        # 不同机制：聊天要的是时序连续（「那它呢」也接得上主语），检索要的是
        # 跨会话命中（去年的同类问题也能被翻出来）。只有 memory 没有 history
        # 的聊天服务，第二句就失忆 —— 这正是 HTTP 层需要的补丁。
        if history:
            messages.extend(history)
        messages.append(user_message(user_input))

        yield tracer.emit(
            EventType.AGENT_START,
            input=user_input,
            tools=list(self.registry.names),
            max_steps=config.max_steps,
            recalled=len(recalled),
        )
        if recalled:
            yield tracer.emit(EventType.MEMORY_RECALL, count=len(recalled))

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
                    yield tracer.emit(
                        EventType.CONTEXT_TRIMMED, step=step, **context.stats.to_dict()
                    )
            request = GenerateRequest(
                messages=request_messages,
                tools=self.registry.specs(),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )

            with tracer.span(
                EventType.LLM_CALL, step=step, message_count=len(messages)
            ) as call_event:
                yield call_event
                response: LLMResponse = await self._generate(request)
            usage = usage + response.usage

            assistant_msg = response.message
            messages.append(assistant_msg)
            yield tracer.emit(
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
                yield tracer.emit(
                    EventType.TOOL_CALL,
                    step=step,
                    tool=call.name,
                    arguments=call.arguments,
                )
                result = await self.registry.ainvoke(call)
                yield tracer.emit(
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
                    yield tracer.emit(
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
            yield tracer.emit(EventType.MAX_STEPS, step=step, max_steps=config.max_steps)

        yield tracer.emit(
            EventType.AGENT_END,
            step=step,
            stopped_reason=stopped.value,
            total_tokens=usage.total_tokens,
        )

        if memory is not None and answer:
            memory.remember_turn(user_input, answer, tags=(self.name,))

        yield AgentResult(
            answer=answer,
            stopped_reason=stopped,
            steps=step,
            messages=tuple(messages),
            usage=usage,
            trace=tuple(tracer.events),
        )

    # ------------------------------------------------------------ 内部

    async def _generate(self, request: GenerateRequest) -> LLMResponse:
        """One model call, preferring a native async path when available."""
        agenerate = getattr(self.provider, "agenerate", None)
        if callable(agenerate):
            result = agenerate(request)
            if inspect.isawaitable(result):
                return await result
        # 同步 provider：进线程池。就地调用会占住事件循环，
        # 一个 30 秒的模型请求足以让整个服务失去响应。
        return await asyncio.to_thread(self.provider.generate, request)


def _last_assistant_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role.value == "assistant" and message.content:
            return message.content
    return ""


def _require_result(result: AgentResult | None) -> AgentResult:
    """Narrow ``AgentResult | None`` — and fail loudly if the contract broke.

    ``_run_stream`` 的最后一项一定是结果。这里不做 ``assert``：隐藏契约一旦
    被后来的改动破坏，调用方需要的是一条能看懂的错，而不是半个 ``None``。
    """
    if result is None:
        raise AgentError("agent stream ended without a result")
    return result
