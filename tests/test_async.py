"""Async surface tests — still fully offline.

这里要证明的不是「有个 arun 方法」这种表面事实，而是三条容易被糊弄过去
的性质：

1. **真的没阻塞事件循环**。一个只在 async 函数里写 ``time.sleep`` 的实现
   也能骗过「await 能跑通」的测试。所以这里用时间线断言：慢 IO 期间，
   另一个协程必须先完成。
2. **同步实现与异步实现行为一致**。两条路径跑同一个脚本，结果必须逐条相等。
3. **错误路径也是路径**。async 工具抛异常、同步入口拿到 awaitable、
   事件循环里调 run()，都要给出明确结果而不是静默降级。
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from conftest import add
from nexus_agents import (
    CompiledGraph,
    GraphError,
    InMemoryCheckpointer,
    ReActAgent,
    StateGraph,
    ToolRegistry,
    build_tool_spec,
    tool,
)
from nexus_agents.checkpoint import Checkpoint
from nexus_agents.errors import ThreadNotFoundError
from nexus_agents.providers import (
    OpenAICompatibleProvider,
    ScriptedProvider,
    tool_call_message,
)
from nexus_agents.providers.base import GenerateRequest, LLMResponse, supports_async
from nexus_agents.types import ToolCall, assistant_message, user_message

BASE = "https://llm.test/v1"


# ================================================================ 测试替身


@tool
async def fetch_price(name: str) -> str:
    """Look up a price without blocking the loop.

    Args:
        name: product name
    """
    await asyncio.sleep(0)
    return f"{name}=128"


@tool
def blocking_lookup(name: str) -> str:
    """A synchronous tool that would block if it ran inline.

    Args:
        name: product name
    """
    return f"{name}=ok"


@tool
async def explode() -> str:
    """Always fails."""
    raise RuntimeError("boom")


class SyncOnlyProvider:
    """A provider written against the sync contract only."""

    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.threads: list[str] = []

    def generate(self, request: GenerateRequest) -> LLMResponse:
        self.threads.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        return LLMResponse(message=assistant_message("sync-done"))


class BothProvider(SyncOnlyProvider):
    """Implements both paths so we can see which one gets picked."""

    def __init__(self) -> None:
        super().__init__()
        self.agenerate_calls = 0

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.agenerate_calls += 1
        return LLMResponse(message=assistant_message("async-done"))


def chat_payload(content: str = "hi") -> dict:
    message = {"role": "assistant", "content": content}
    return {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
    }


def http_client(handler) -> httpx.Client:
    return httpx.Client(base_url=BASE, transport=httpx.MockTransport(handler))


def async_http_client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=BASE, transport=httpx.MockTransport(handler))


def make_request() -> GenerateRequest:
    return GenerateRequest(messages=(user_message("hello"),))


# ================================================================ 工具层


def test_async_tools_are_detected():
    assert fetch_price.is_async is True
    assert blocking_lookup.is_async is False


@pytest.mark.asyncio
async def test_ainvoke_runs_an_async_tool():
    result = await fetch_price.ainvoke(
        ToolCall(id="c1", name="fetch_price", arguments={"name": "cup"})
    )

    assert result.is_error is False
    assert result.output == "cup=128"


def test_sync_invoke_refuses_an_async_tool():
    """同步路径不能 await，与其偷偷 asyncio.run，不如明确报错。"""
    result = fetch_price.invoke(name="cup")

    assert result.is_error is True
    assert "async" in result.error
    assert "arun" in result.error


@pytest.mark.asyncio
async def test_ainvoke_runs_sync_tools_off_the_main_thread():
    seen: dict[str, str] = {}

    @tool
    def where() -> str:
        """Report which thread executed this tool."""
        seen["thread"] = threading.current_thread().name
        return "ok"

    await where.ainvoke()

    assert seen["thread"] != threading.main_thread().name


def test_sync_invoke_rejects_a_returned_awaitable():
    """被装饰器包过的 async 函数：is_async 漏判，但结果仍是 awaitable。"""

    async def inner() -> str:
        return "x"

    def wrapper() -> str:
        return inner()  # type: ignore[return-value]

    spec = build_tool_spec(wrapper, name="wrapper")
    result = spec.invoke()

    assert result.is_error is True
    assert "awaitable" in result.error


@pytest.mark.asyncio
async def test_async_tool_errors_become_tool_results():
    result = await explode.ainvoke()

    assert result.is_error is True
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_registry_ainvoke_reports_unknown_tools():
    registry = ToolRegistry([fetch_price])

    result = await registry.ainvoke(ToolCall(id="c", name="nope", arguments={}))

    assert result.is_error is True
    assert "nope" in result.error


@pytest.mark.asyncio
async def test_registry_ainvoke_dispatches_to_the_right_tool():
    registry = ToolRegistry([fetch_price, blocking_lookup])

    result = await registry.ainvoke(
        ToolCall(id="c", name="blocking_lookup", arguments={"name": "kiln"})
    )

    assert result.output == "kiln=ok"


# ================================================================ Agent


@pytest.mark.asyncio
async def test_arun_completes_with_an_async_tool():
    provider = ScriptedProvider(
        [tool_call_message("fetch_price", {"name": "cup"}), "the cup is 128"]
    )
    agent = ReActAgent(provider, [fetch_price])

    result = await agent.arun("how much is the cup?")

    assert result.answer == "the cup is 128"
    assert result.steps == 2
    assert result.succeeded is True
    tool_messages = [m for m in result.messages if m.role.value == "tool"]
    assert tool_messages[0].content == "cup=128"


def test_run_and_arun_produce_the_same_result():
    """同一个脚本、同一个 agent 定义，两条路径必须给出同一份结果。"""

    def build() -> ReActAgent:
        return ReActAgent(
            ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"]), [add]
        )

    sync_result = build().run("2+3")
    async_result = asyncio.run(build().arun("2+3"))

    assert sync_result.answer == async_result.answer == "5"
    assert sync_result.steps == async_result.steps
    assert [m.role for m in sync_result.messages] == [
        m.role for m in async_result.messages
    ]
    assert sync_result.usage.total_tokens == async_result.usage.total_tokens


@pytest.mark.asyncio
async def test_run_refuses_to_nest_event_loops():
    agent = ReActAgent(ScriptedProvider(["hi"]))

    with pytest.raises(RuntimeError, match="arun"):
        agent.run("hello")


@pytest.mark.asyncio
async def test_sync_provider_is_called_off_the_main_thread():
    provider = SyncOnlyProvider()
    agent = ReActAgent(provider)

    result = await agent.arun("hi")

    assert result.answer == "sync-done"
    assert len(provider.threads) == 1
    assert provider.threads[0] != threading.main_thread().name


@pytest.mark.asyncio
async def test_native_async_provider_is_preferred():
    provider = BothProvider()

    result = await ReActAgent(provider).arun("hi")

    assert result.answer == "async-done"
    assert provider.agenerate_calls == 1
    assert provider.threads == []  # generate() 一次都没被调用


@pytest.mark.asyncio
async def test_a_slow_sync_provider_does_not_stall_the_loop():
    """时间线断言：慢 provider 在跑的时候，另一个协程必须先跑完。

    如果实现是「在 async 函数里直接调同步方法」，ticker 会被延迟到 agent
    结束之后，顺序就反了。
    """
    timeline: list[str] = []
    agent = ReActAgent(SyncOnlyProvider(delay=0.08))

    async def run_agent() -> None:
        await agent.arun("hi")
        timeline.append("agent")

    async def ticker() -> None:
        await asyncio.sleep(0.01)
        timeline.append("tick")

    await asyncio.gather(run_agent(), ticker())

    assert timeline == ["tick", "agent"]


@pytest.mark.asyncio
async def test_a_slow_sync_tool_does_not_stall_the_loop():
    timeline: list[str] = []

    @tool
    def slow_tool() -> str:
        """Pretend to do blocking IO on the calling thread."""
        time.sleep(0.08)
        return "slow"

    provider = ScriptedProvider([tool_call_message("slow_tool"), "done"])
    agent = ReActAgent(provider, [slow_tool])

    async def run_agent() -> None:
        await agent.arun("go")
        timeline.append("agent")

    async def ticker() -> None:
        await asyncio.sleep(0.01)
        timeline.append("tick")

    await asyncio.gather(run_agent(), ticker())

    assert timeline == ["tick", "agent"]


@pytest.mark.asyncio
async def test_two_agents_can_run_concurrently():
    """两个 arun 交替推进，而不是一个跑完再跑另一个。"""

    class CountingProvider:
        def __init__(self) -> None:
            self.in_flight = 0
            self.peak = 0

        async def agenerate(self, request: GenerateRequest) -> LLMResponse:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            await asyncio.sleep(0.02)
            self.in_flight -= 1
            return LLMResponse(message=assistant_message("ok"))

    provider = CountingProvider()
    agent = ReActAgent(provider)

    await asyncio.gather(agent.arun("a"), agent.arun("b"), agent.arun("c"))

    assert provider.peak == 3


@pytest.mark.asyncio
async def test_arun_keeps_the_full_history_like_run():
    provider = ScriptedProvider([tool_call_message("add", {"a": 1, "b": 1}), "2"])
    agent = ReActAgent(provider, [add])

    result = await agent.arun("1+1")

    assert [m.role.value for m in result.messages] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    from nexus_agents.tracing import EventType

    assert result.trace[-1].type == EventType.AGENT_END


# ================================================================ 图


def linear_graph(node) -> CompiledGraph:
    graph = StateGraph()
    graph.add_node("only", node)
    graph.set_entry_point("only")
    return graph.compile()


@pytest.mark.asyncio
async def test_graph_ainvoke_awaits_async_nodes():
    async def bump(state):
        await asyncio.sleep(0)
        return {"value": state.get("value", 0) + 1}

    result = await linear_graph(bump).ainvoke({"value": 1})

    assert result.state["value"] == 2
    assert result.path == ("only",)


def test_graph_invoke_also_handles_async_nodes():
    """同步入口开自己的一次性 loop，async 节点照常被 await。"""

    async def bump(state):
        return {"value": state.get("value", 0) + 1}

    result = linear_graph(bump).invoke({"value": 41})

    assert result.state["value"] == 42


@pytest.mark.asyncio
async def test_graph_invoke_refuses_to_nest_event_loops():
    compiled = linear_graph(lambda state: {})

    with pytest.raises(RuntimeError, match="ainvoke"):
        compiled.invoke({})


@pytest.mark.asyncio
async def test_graph_async_router_is_rejected_with_a_clear_error():
    async def router(state):
        return "yes"

    graph = StateGraph()
    graph.add_node("start", lambda state: {})
    graph.add_node("done", lambda state: {})
    graph.set_entry_point("start")
    graph.add_conditional_edges("start", router, {"yes": "done"})
    compiled = graph.compile()

    with pytest.raises(GraphError, match="synchronous"):
        await compiled.ainvoke({})


@pytest.mark.asyncio
async def test_graph_aresume_continues_from_a_checkpoint():
    async def first(state):
        return {"log": "a"}

    async def second(state):
        return {"log": "b"}

    store = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("a", first)
    graph.add_node("b", second)
    graph.add_edge("a", "b")
    graph.set_entry_point("a")
    compiled = graph.compile(checkpointer=store)

    store.save(Checkpoint(thread_id="t1", step=1, node="b", state={"log": "a"}))
    resumed = await compiled.aresume("t1")

    assert resumed.path == ("b",)
    assert resumed.state["log"] == "b"


@pytest.mark.asyncio
async def test_aresume_reports_unknown_threads():
    store = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("n", lambda state: {})
    graph.set_entry_point("n")
    compiled = graph.compile(checkpointer=store)

    with pytest.raises(ThreadNotFoundError):
        await compiled.aresume("missing")


@pytest.mark.asyncio
async def test_aresume_requires_a_checkpointer():
    compiled = linear_graph(lambda state: {})

    with pytest.raises(Exception, match="checkpointer"):
        await compiled.aresume("nope")


# ================================================================ Provider


def test_supports_async_detects_native_methods():
    assert supports_async(ScriptedProvider(["x"])) is True
    assert supports_async(SimpleNamespace(generate=lambda r: None)) is False


@pytest.mark.asyncio
async def test_agenerate_parses_a_normal_response():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_payload("hello there"))

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url=BASE,
        client=http_client(handler),
        async_client=async_http_client(handler),
    )

    response = await provider.agenerate(make_request())

    assert response.message.content == "hello there"
    assert response.usage.total_tokens == 5


@pytest.mark.asyncio
async def test_agenerate_retries_retryable_statuses():
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=chat_payload("recovered"))

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url=BASE,
        client=http_client(handler),
        async_client=async_http_client(handler),
        max_retries=1,
        backoff_base=0.0,
    )

    response = await provider.agenerate(make_request())

    assert response.message.content == "recovered"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_agenerate_surfaces_http_errors():
    from nexus_agents.errors import ProviderTransportError

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url=BASE,
        client=http_client(handler),
        async_client=async_http_client(handler),
        max_retries=0,
    )

    with pytest.raises(ProviderTransportError, match="401"):
        await provider.agenerate(make_request())


@pytest.mark.asyncio
async def test_aclose_is_safe_without_an_async_client():
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=chat_payload())

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url=BASE,
        client=http_client(handler),
    )

    await provider.aclose()  # 从未建过 async client，不应报错


@pytest.mark.asyncio
async def test_lazy_async_client_is_created_and_released():
    """不注入 async client 时按需建一个，用完能关掉。

    用不可达地址（127.0.0.1:9）而不是真实 API：这里要验证的是「客户端被建了」
    和「连接失败被正确包装」，不是网络本身。
    """
    from nexus_agents.errors import ProviderTransportError

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url="http://127.0.0.1:9/v1",
        timeout=0.5,
        max_retries=0,
    )

    with pytest.raises(ProviderTransportError):
        await provider.agenerate(make_request())

    assert provider._async_client is not None
    await provider.aclose()
    assert provider._async_client is None


@pytest.mark.asyncio
async def test_agenerate_raises_on_a_non_json_body():
    from nexus_agents.errors import ProviderResponseError

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>not json</html>")

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url=BASE,
        client=http_client(handler),
        async_client=async_http_client(handler),
    )

    with pytest.raises(ProviderResponseError, match="JSON"):
        await provider.agenerate(make_request())


@pytest.mark.asyncio
async def test_agenerate_gives_up_after_exhausting_retries():
    from nexus_agents.errors import ProviderTransportError

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="still down")

    provider = OpenAICompatibleProvider(
        model="test-model",
        base_url=BASE,
        client=http_client(handler),
        async_client=async_http_client(handler),
        max_retries=1,
        backoff_base=0.0,
    )

    with pytest.raises(ProviderTransportError, match="503"):
        await provider.agenerate(make_request())


@pytest.mark.asyncio
async def test_ainvoke_awaits_a_tool_that_returns_an_awaitable():
    """同步函数体内返回 coroutine 的情况（被装饰器包过的 async 函数）。"""

    async def inner() -> str:
        return "unwrapped"

    def wrapper() -> str:
        return inner()  # type: ignore[return-value]

    spec = build_tool_spec(wrapper, name="wrapper")

    result = await spec.ainvoke()

    assert result.is_error is False
    assert result.output == "unwrapped"


@pytest.mark.asyncio
async def test_graph_awaits_a_node_that_returns_an_awaitable():
    async def inner() -> dict:
        return {"value": 7}

    def wrapper(state):
        return inner()  # type: ignore[return-value]

    result = await linear_graph(wrapper).ainvoke({})

    assert result.state["value"] == 7
