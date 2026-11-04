"""Async surface demo — runs offline, no API key, no network.

三个演示，每个对应一个「异步到底值不值」的具体问题：

1. ``run`` 与 ``arun`` 是否给出同一个结果（正确性）
2. 三个 agent 并发是不是真的更快（吞吐）
3. 一个阻塞的工具会不会把事件循环一起卡死（可靠性）

第三条是最容易被忽略的：很多「异步」实现只是把 async 关键字贴上去，
内部照样同步调用，单测全过，一上生产就整服务卡住。
"""

from __future__ import annotations

import asyncio
import time

from nexus_agents import ReActAgent, tool
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.types import assistant_message

# ---------------------------------------------------------------- 工具

@tool
async def fetch_price(name: str) -> str:
    """Look up a price asynchronously.

    Args:
        name: product name
    """
    await asyncio.sleep(0.02)
    return f"{name}: 128 CNY"


@tool
def slow_report(name: str) -> str:
    """A blocking tool — the kind that silently ruins an async server.

    Args:
        name: product name
    """
    time.sleep(0.05)
    return f"{name}: report ready"


class SlowProvider:
    """A deliberately slow provider, so concurrency is visible."""

    def __init__(self, delay: float = 0.08) -> None:
        self.delay = delay
        self.calls = 0

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.calls += 1
        await asyncio.sleep(self.delay)
        return LLMResponse(message=assistant_message("done"))


# ---------------------------------------------------------------- 1. 结果一致

def same_result() -> None:
    def build() -> ReActAgent:
        return ReActAgent(
            ScriptedProvider([tool_call_message("fetch_price", {"name": "teacup"}), "128"]),
            [fetch_price],
        )

    sync_result = build().run("how much is a teacup?")
    async_result = asyncio.run(build().arun("how much is a teacup?"))

    sync_roles = [m.role.value for m in sync_result.messages]
    async_roles = [m.role.value for m in async_result.messages]

    print("=== 1. run() and arun() agree ===")
    print(f"run()   answer={sync_result.answer!r} steps={sync_result.steps}")
    print(f"arun()  answer={async_result.answer!r} steps={async_result.steps}")
    print(f"same message roles: {sync_roles == async_roles}")
    print()


# ---------------------------------------------------------------- 2. 并发

async def concurrency() -> None:
    agent = ReActAgent(SlowProvider(delay=0.08))

    started = time.perf_counter()
    for _ in range(3):
        await agent.arun("hi")
    serial = time.perf_counter() - started

    started = time.perf_counter()
    await asyncio.gather(*(agent.arun("hi") for _ in range(3)))
    concurrent = time.perf_counter() - started

    print("=== 2. three agent runs ===")
    print(f"one by one   {serial * 1000:6.0f} ms")
    print(f"gathered     {concurrent * 1000:6.0f} ms")
    print(f"speed-up     {serial / concurrent:6.2f}x   (provider delay 80 ms)")
    print()


# ---------------------------------------------------------------- 3. 不阻塞

async def no_stall() -> None:
    timeline: list[str] = []
    agent = ReActAgent(
        ScriptedProvider(
            [tool_call_message("slow_report", {"name": "B7"}), "report is ready"]
        ),
        [slow_report],
    )

    async def run_agent() -> None:
        await agent.arun("build the report")
        timeline.append("agent finished")

    async def ticker() -> None:
        await asyncio.sleep(0.01)
        timeline.append("ticker woke up (loop still alive)")

    await asyncio.gather(run_agent(), ticker())

    print("=== 3. a blocking tool does not freeze the loop ===")
    print("the tool calls time.sleep(0.05) — blocking on purpose")
    for item in timeline:
        print(f"  {item}")
    print()


def main() -> None:
    same_result()
    asyncio.run(concurrency())
    asyncio.run(no_stall())
    print("all demos ran offline — no API key, no network")


if __name__ == "__main__":
    main()
