"""Streaming tests — offline, and deliberately hostile.

流式最容易糊弄过去。把 ``arun`` 包一层「跑完再逐条 yield」也能让
``async for`` 跑通，测试全绿，上线后用户仍然盯着空白页等 40 秒。

所以这里的第一条测试不看结果，只看**时间线**：``llm.call`` 必须先于模型
返回到达消费者，消费者松手之后模型才允许结束。剩下几条钉住契约：
事件与 trace 一致、最后一项是结果、中途放弃不算一轮问答、错误照常抛出、
同一个 agent 上的两条流互不干扰。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from conftest import add
from nexus_agents import AgentResult, ReActAgent, tool
from nexus_agents.agent import AgentStreamItem
from nexus_agents.errors import AgentError
from nexus_agents.memory import ConversationMemory
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.tracing import EventType, TraceEvent
from nexus_agents.types import Usage, assistant_message

# ================================================================ 测试替身


class GatedProvider:
    """Finishes only when the test opens the gate.

    用来证明 ``astream`` 不是「跑完再回放」：只要模型还卡在 ``release`` 上，
    事件就必须已经到达消费者手里。
    """

    def __init__(self, reply: str = "late answer") -> None:
        self.reply = reply
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0
        self.finished = False

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.calls += 1
        self.started.set()
        await self.release.wait()
        self.finished = True
        return LLMResponse(
            message=assistant_message(self.reply),
            usage=Usage(prompt_tokens=3, completion_tokens=4),
        )


class ExplodingProvider:
    """A backend that dies mid-run."""

    def __init__(self) -> None:
        self.calls = 0

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.calls += 1
        raise RuntimeError("upstream is on fire")


@tool
def slow_lookup(name: str) -> str:
    """Look up a name without blocking the event loop.

    Args:
        name: what to look up
    """
    return f"{name}=42"


async def _collect(stream: AsyncIterator[AgentStreamItem]) -> list[AgentStreamItem]:
    return [item async for item in stream]


def _events(items: list[AgentStreamItem]) -> list[TraceEvent]:
    return [item for item in items if isinstance(item, TraceEvent)]


def _final(items: list[AgentStreamItem]) -> AgentResult:
    last = items[-1]
    assert isinstance(last, AgentResult)
    return last


# ================================================================ 真增量


@pytest.mark.asyncio
async def test_llm_call_arrives_before_the_model_finishes():
    provider = GatedProvider()
    stream = ReActAgent(provider).astream("hello")

    first = await anext(stream)
    assert isinstance(first, TraceEvent)
    assert first.type == EventType.AGENT_START

    second = await anext(stream)
    assert isinstance(second, TraceEvent)
    assert second.type == EventType.LLM_CALL
    # 调用刚发起：既没有结果，也还没有耗时
    assert second.duration_ms is None
    assert not provider.finished

    # 恢复生成器 -> 真正开始调模型；模型卡在自己的门后面
    pending = asyncio.create_task(anext(stream))
    await asyncio.wait_for(provider.started.wait(), timeout=5.0)
    assert not pending.done(), "llm.response 不该在模型返回前出现"

    provider.release.set()
    third = await pending
    assert isinstance(third, TraceEvent)
    assert third.type == EventType.LLM_RESPONSE

    result = _final(await _collect(stream))
    assert result.answer == "late answer"
    assert provider.finished


@pytest.mark.asyncio
async def test_tool_events_are_streamed_as_the_tool_runs():
    """工具执行前后都要有事件，否则界面只能显示「思考中」。"""
    provider = ScriptedProvider([tool_call_message("slow_lookup", {"name": "widget"}), "42"])
    items = await _collect(ReActAgent(provider, [slow_lookup]).astream("price?"))

    types = [item.type for item in _events(items)]
    assert types.index(EventType.TOOL_CALL) < types.index(EventType.TOOL_RESULT)
    call_event = _events(items)[types.index(EventType.TOOL_CALL)]
    assert call_event.payload["tool"] == "slow_lookup"
    assert call_event.payload["arguments"] == {"name": "widget"}
    assert _final(items).answer == "42"


# ================================================================ 契约


@pytest.mark.asyncio
async def test_the_last_item_is_always_the_result():
    provider = ScriptedProvider(["done"])
    items = await _collect(ReActAgent(provider).astream("hi"))

    assert sum(isinstance(item, AgentResult) for item in items) == 1
    assert isinstance(items[-1], AgentResult)
    assert items[-1].answer == "done"


@pytest.mark.asyncio
async def test_streamed_events_are_the_result_trace():
    """边看边流和跑完再看，必须是同一份事实，不能各自维护一套。"""
    provider = ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"])
    items = await _collect(ReActAgent(provider, [add]).astream("sum"))

    result = _final(items)
    assert _events(items) == list(result.trace)
    types = [e.type for e in result.trace]
    assert types[0] == EventType.AGENT_START
    assert types[-1] == EventType.AGENT_END


@pytest.mark.asyncio
async def test_arun_is_the_drained_stream():
    direct = await ReActAgent(
        ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"]), [add]
    ).arun("sum")
    streamed = _final(
        await _collect(
            ReActAgent(
                ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"]), [add]
            ).astream("sum")
        )
    )

    assert direct.answer == streamed.answer
    assert direct.steps == streamed.steps
    assert direct.messages == streamed.messages
    assert [e.type for e in direct.trace] == [e.type for e in streamed.trace]
    assert direct.usage.total_tokens == streamed.usage.total_tokens


def test_sync_run_matches_the_streamed_result():
    sync_result = ReActAgent(ScriptedProvider(["same"])).run("hi")
    streamed = asyncio.run(_collect(ReActAgent(ScriptedProvider(["same"])).astream("hi")))[-1]

    assert isinstance(streamed, AgentResult)
    assert sync_result.answer == streamed.answer
    assert [e.type for e in sync_result.trace] == [e.type for e in streamed.trace]


# ================================================================ 放弃与记忆


@pytest.mark.asyncio
async def test_abandoning_a_stream_does_not_write_memory():
    """半途而废的一轮不该变成记忆 —— 那会让下次召回出现没发生过的对话。"""
    memory = ConversationMemory()
    stream = ReActAgent(ScriptedProvider(["never delivered"])).astream("question", memory=memory)

    await anext(stream)
    await stream.aclose()

    assert len(memory.store) == 0


@pytest.mark.asyncio
async def test_a_fully_consumed_stream_writes_memory():
    memory = ConversationMemory()
    items = await _collect(
        ReActAgent(ScriptedProvider(["stored answer"])).astream("question", memory=memory)
    )

    assert _final(items).answer == "stored answer"
    assert len(memory.store) == 1
    record = memory.store.all()[0]
    assert "stored answer" in record.text
    assert record.tags == frozenset({"agent"})


@pytest.mark.asyncio
async def test_a_stream_can_be_abandoned_before_the_first_llm_call():
    """在 agent.start 处就松手：既不能崩，也不能写记忆。"""
    memory = ConversationMemory()
    stream = ReActAgent(ScriptedProvider(["x"])).astream("q", memory=memory)

    await anext(stream)
    await stream.aclose()
    await stream.aclose()  # 重复关闭是安全的

    assert len(memory.store) == 0


# ================================================================ 失败路径


@pytest.mark.asyncio
async def test_provider_errors_propagate_out_of_the_stream():
    provider = ExplodingProvider()
    stream = ReActAgent(provider).astream("hi")

    await anext(stream)  # agent.start
    await anext(stream)  # llm.call
    with pytest.raises(RuntimeError, match="on fire"):
        await anext(stream)
    assert provider.calls == 1


@pytest.mark.asyncio
async def test_an_event_only_stream_is_rejected_by_arun(monkeypatch: Any):
    """内部契约被破坏时要明确报错，而不是返回 None 让调用方在远处崩。"""
    agent = ReActAgent(ScriptedProvider(["x"]))

    async def events_only(*args: Any, **kwargs: Any) -> AsyncIterator[AgentStreamItem]:
        for item in ():
            yield item

    monkeypatch.setattr(agent, "_run_stream", events_only)

    with pytest.raises(AgentError, match="without a result"):
        await agent.arun("hi")


# ================================================================ 并发


@pytest.mark.asyncio
async def test_two_streams_from_one_agent_do_not_share_state():
    """状态全在生成器里，``self`` 上没有「当前运行」这种字段。"""
    agent = ReActAgent(ScriptedProvider(["same answer"]))

    async def consume(prompt: str) -> list[AgentStreamItem]:
        return await _collect(agent.astream(prompt))

    left, right = await asyncio.gather(consume("left"), consume("right"))

    assert _final(left).answer == _final(right).answer == "same answer"
    assert _events(left)[0].payload["input"] == "left"
    assert _events(right)[0].payload["input"] == "right"
