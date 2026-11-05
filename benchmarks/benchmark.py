"""Offline performance benchmarks — no API key, no network, no server.

为什么基准测试也要离线
----------------------
用可编程延迟的 provider 和阻塞工具，测的是**运行时自身的开销**：事件调度、
线程池分派、会话锁、SSE 帧编码。真实 LLM 的延迟（几百毫秒到几十秒）会把
这些数字完全淹没；在共享 CI 机器上跑真实网络调用只会得到噪声。

怎么读这些数字
--------------
1. 每轮开销：provider 立即返回，每轮延迟 ≈ 框架纯开销（预期亚毫秒级）。
2. 并发：10 个约 170ms 的运行并发执行，墙上时间应接近单轮，而不是 10 倍。
   工具是刻意的同步 ``time.sleep`` —— 验证同步工具真的进了线程池。
3. 流式首帧：``astream`` 的第一个事件应几乎立即到达，用户不必等整轮结束。
4. HTTP 层：进程内走 ASGI 调 ``create_app``，量化 FastAPI 请求解析 + 会话锁
   + 历史拼接让一轮变贵了多少。

运行：``python benchmarks/benchmark.py``（约 10 秒）。
"""

from __future__ import annotations

import asyncio
import math
import os
import platform
import time

import httpx

from nexus_agents import ReActAgent, tool
from nexus_agents.providers import tool_call_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.service import ServiceSettings, create_app
from nexus_agents.types import Role, Usage, assistant_message

# ---------------------------------------------------------------- 测量工具


def percentile(samples: list[float], pct: float) -> float:
    """Nearest-rank percentile. Input in milliseconds, output in milliseconds."""
    ordered = sorted(samples)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[rank - 1]


def stats(samples: list[float]) -> dict[str, float]:
    """P50 / P95 / P99 / max of a list of millisecond samples."""
    return {
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "p99": percentile(samples, 99),
        "max": max(samples),
    }


def report(title: str, numbers: dict[str, float]) -> None:
    print(f"=== {title} ===")
    for name, value in numbers.items():
        # speed-up 是倍数，不是毫秒 —— 单位标错比不标更糟
        unit = "x" if "speed-up" in name else "ms"
        print(f"  {name:<24} {value:10.2f} {unit}")
    print()


# ---------------------------------------------------------------- 测试替身


@tool
def echo(text: str) -> str:
    """Return the input unchanged.

    Args:
        text: anything
    """
    return text


@tool
def blocking_lookup(keyword: str) -> str:
    """A blocking tool on purpose — verifies sync tools really hit the thread pool.

    Args:
        keyword: what to look up
    """
    time.sleep(0.05)
    return f"{keyword}: 12"


class DelayedProvider:
    """Scripted two-step loop (tool call, then answer) with an artificial delay.

    分不分「第几次调用」有讲究：用全局计数器交替的话，多个会话交错请求时
    奇偶会被打乱 —— 有的一轮只跑一次就结束，基准数字看起来更漂亮但全是
    假的。真实模型是**看上下文**决定行为的：上一条消息还是工具结果就给
    最终答案，否则先要工具。这里照此实现，延迟用 ``delay`` 模拟网络。
    """

    def __init__(self, delay: float, tool_name: str, tool_args: dict[str, str]) -> None:
        self.delay = delay
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.calls = 0

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.calls += 1
        await asyncio.sleep(self.delay)
        assert request.messages, "provider always sees a non-empty conversation"
        last = request.messages[-1]
        if last.role == Role.TOOL:
            return LLMResponse(
                message=assistant_message("done"),
                usage=Usage(prompt_tokens=5, completion_tokens=5),
            )
        return LLMResponse(message=tool_call_message(self.tool_name, self.tool_args))


def build_agent(delay: float, *, blocking: bool = True) -> ReActAgent:
    """One round = two provider calls + one tool call in between.

    ``blocking=False`` 用立即返回的 ``echo``，测的是纯框架开销；
    ``blocking=True`` 用 ``time.sleep`` 的阻塞工具，测的是并发与线程池分派。
    """
    if blocking:
        return ReActAgent(
            DelayedProvider(delay, "blocking_lookup", {"keyword": "cup"}),
            [blocking_lookup],
            name="bench",
        )
    return ReActAgent(
        DelayedProvider(delay, "echo", {"text": "cup"}), [echo], name="bench"
    )


# ---------------------------------------------------------------- 1. 每轮开销


async def runtime_overhead(rounds: int = 200) -> dict[str, float]:
    """Framework-only cost of one tool-calling round (provider answers instantly)."""
    samples: list[float] = []
    for _ in range(rounds):
        agent = build_agent(delay=0.0, blocking=False)
        started = time.perf_counter()
        result = await agent.arun("check stock")
        samples.append((time.perf_counter() - started) * 1000)
        assert result.succeeded
    return stats(samples)


# ---------------------------------------------------------------- 2. 并发


async def concurrency(rounds: int = 10) -> dict[str, float]:
    """Serial vs gathered runs; each round is ~170ms (60+50+60 ms)."""
    delay = 0.06
    started = time.perf_counter()
    for _ in range(rounds):
        await build_agent(delay=delay).arun("check stock")
    serial_ms = (time.perf_counter() - started) * 1000

    # 下限护栏：串行总时长必须 ≥ rounds × (两次 provider 延迟 + 工具睡眠)。
    # 没有它，"工具调用参数写错 → 校验阶段 0ms 失败"这种静默 bug 会让
    # 基准照常出数字，只是全部变小 —— 数据看着更漂亮，全是假的。
    assert serial_ms >= rounds * (2 * delay + 0.045) * 1000

    started = time.perf_counter()
    runs = [build_agent(delay=delay).arun("check stock") for _ in range(rounds)]
    results = await asyncio.gather(*runs)
    concurrent_ms = (time.perf_counter() - started) * 1000
    assert all(r.succeeded for r in results)

    return {
        f"serial x{rounds}": serial_ms,
        f"concurrent x{rounds}": concurrent_ms,
        "speed-up": serial_ms / concurrent_ms,
    }


# ---------------------------------------------------------------- 3. 流式首帧


async def stream_first_event(rounds: int = 10) -> dict[str, float]:
    """Time to the first astream event vs time to the final AgentResult.

    首帧意义：聊天前端能在第一毫秒就画出一个「正在思考」的进度条，
    而不是等整轮结束才从黑屏里蹦出答案。
    """
    ttff: list[float] = []
    total: list[float] = []
    for _ in range(rounds):
        agent = build_agent(delay=0.08)
        started = time.perf_counter()
        first_seen: float | None = None
        final: float | None = None
        async for _item in agent.astream("check stock"):
            now = (time.perf_counter() - started) * 1000
            if first_seen is None:
                first_seen = now
            final = now
        assert first_seen is not None and final is not None
        ttff.append(first_seen)
        total.append(final)
    merged = stats(ttff)
    merged.update({f"total_{k}": v for k, v in stats(total).items()})
    return merged


# ---------------------------------------------------------------- 4. HTTP 层


async def service_layer(rounds: int = 100, sessions: int = 10) -> dict[str, float]:
    """End-to-end through create_app, in-process via ASGI transport."""
    app = create_app(
        build_agent(delay=0.0, blocking=False),
        settings=ServiceSettings(heartbeat_seconds=5.0),
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:
        session_id = (await client.post("/sessions", json={})).json()["session_id"]
        samples: list[float] = []
        for _ in range(rounds):
            started = time.perf_counter()
            response = await client.post(
                f"/sessions/{session_id}/messages", json={"text": "check stock"}
            )
            samples.append((time.perf_counter() - started) * 1000)
            assert response.status_code == 200
        json_stats = stats(samples)

        # 并发会话：慢 app（每轮 ~170ms），10 个会话同时说话 vs 逐个说
        slow_app = create_app(build_agent(delay=0.06), settings=ServiceSettings())
        slow = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=slow_app), base_url="http://bench-slow"
        )
        try:
            slow_ids = [
                (await slow.post("/sessions", json={})).json()["session_id"]
                for _ in range(sessions)
            ]
            started = time.perf_counter()
            for sid in slow_ids:
                await slow.post(f"/sessions/{sid}/messages", json={"text": "check stock"})
            serial_ms = (time.perf_counter() - started) * 1000
            # 同 concurrency() 的护栏：慢路径必须真的慢
            assert serial_ms >= sessions * (2 * 0.06 + 0.045) * 1000

            started = time.perf_counter()
            replies = await asyncio.gather(
                *(
                    slow.post(f"/sessions/{sid}/messages", json={"text": "again"})
                    for sid in slow_ids
                )
            )
            concurrent_ms = (time.perf_counter() - started) * 1000
            assert all(r.status_code == 200 for r in replies)
        finally:
            await slow.aclose()

    merged = {f"json_{k}": v for k, v in json_stats.items()}
    merged.update(
        {
            f"http_serial x{sessions}": serial_ms,
            f"http_concurrent x{sessions}": concurrent_ms,
            "http_speed-up": serial_ms / concurrent_ms,
        }
    )
    return merged


# ---------------------------------------------------------------- 入口


async def main() -> None:
    print(f"python {platform.python_version()} on {platform.platform()}")
    print(f"cpu_count={os.cpu_count()}")
    print()
    report("1. per-round overhead (agent, no delay)", await runtime_overhead())
    report("2. concurrency (agent, 170ms rounds)", await concurrency())
    report("3. stream time-to-first-event (80ms provider)", await stream_first_event())
    report("4. service layer (in-process ASGI)", await service_layer())
    print("all benchmarks ran offline — no API key, no network, no ports")


if __name__ == "__main__":
    asyncio.run(main())
