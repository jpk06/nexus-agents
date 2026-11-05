"""Streaming demo — runs offline, no API key, no network.

要看的不是「API 长什么样」，而是**事件到达的时刻**。每行前面的 ``+xx ms``
是相对启动的时间，所以能直接看出：

1. ``llm.call`` 在模型返回**之前**就发出去了 —— 这正是进度提示的依据
2. 工具调用与结果之间有真实间隔（这里是 60 ms 的假 IO）
3. 中途 ``break`` 的流不会写记忆

最后一段顺带把事件打成 SSE 的 ``data: ...`` 形状 —— 真正的端点
（``nexus_agents.service``）就是在这个形状上加了会话、心跳与取消，
见 ``examples/service_demo.py``。
"""

from __future__ import annotations

import asyncio
import json
import time

from nexus_agents import AgentResult, ReActAgent, tool
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.types import assistant_message


@tool
def search_inventory(keyword: str) -> str:
    """Search the warehouse for a keyword (slow on purpose).

    Args:
        keyword: what to search for
    """
    time.sleep(0.06)
    return f"{keyword}: 12 items in stock"


class PonderingProvider:
    """Takes 80 ms to answer, and says nothing in the meantime."""

    def __init__(self, delay: float = 0.08) -> None:
        self.delay = delay

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        await asyncio.sleep(self.delay)
        return LLMResponse(message=assistant_message("teacups are in stock"))


# ---------------------------------------------------------------- 1. 实时事件

async def live_timeline() -> None:
    provider = ScriptedProvider(
        [tool_call_message("search_inventory", {"keyword": "teacup"}), "12 件在库"]
    )
    agent = ReActAgent(provider, [search_inventory])

    started = time.perf_counter()

    def stamp() -> str:
        return f"+{(time.perf_counter() - started) * 1000:6.0f} ms"

    print("=== 1. events arrive while the run is still going ===")
    async for item in agent.astream("还有多少茶杯?"):
        if isinstance(item, AgentResult):
            print(f"{stamp()}  <- result: answer={item.answer!r}")
            continue
        summary = ", ".join(f"{k}={v}" for k, v in item.payload.items() if k != "arguments")
        print(f"{stamp()}  {item.type:<14} {summary}")
    print()


# ---------------------------------------------------------------- 2. 慢模型

async def slow_model() -> None:
    agent = ReActAgent(PonderingProvider(delay=0.08))
    started = time.perf_counter()

    print("=== 2. a slow model still shows progress immediately ===")
    async for item in agent.astream("库存情况?"):
        elapsed = (time.perf_counter() - started) * 1000
        if isinstance(item, AgentResult):
            print(f"+{elapsed:6.0f} ms  done: {item.answer!r}")
        else:
            print(f"+{elapsed:6.0f} ms  {item.type}")
    print("   llm.call 出现在 0 ms 附近 —— 用户在模型思考时就知道请求已经发出")
    print()


# ---------------------------------------------------------------- 3. 中途放弃

async def abandon() -> None:
    from nexus_agents.memory import ConversationMemory

    memory = ConversationMemory()
    agent = ReActAgent(PonderingProvider(delay=0.01))
    stream = agent.astream("这条不该被记住", memory=memory)

    print("=== 3. abandoning a stream is a cancellation, not a turn ===")
    await anext(stream)  # agent.start
    print("  received 1 event, then closed early")
    await stream.aclose()
    print(f"  memory records after abort: {len(memory.store)}")
    print()


# ---------------------------------------------------------------- 4. SSE 形状

async def sse_preview() -> None:
    agent = ReActAgent(ScriptedProvider(["hello from the runtime"]))
    print("=== 4. what an SSE endpoint would send ===")
    async for item in agent.astream("hi"):
        if isinstance(item, AgentResult):
            payload = {"type": "agent.result", "answer": item.answer}
        else:
            payload = item.to_dict()
        print(f"data: {json.dumps(payload, ensure_ascii=False)}")
    print()


def main() -> None:
    asyncio.run(live_timeline())
    asyncio.run(slow_model())
    asyncio.run(abandon())
    asyncio.run(sse_preview())
    print("all demos ran offline — no API key, no network")


if __name__ == "__main__":
    main()
