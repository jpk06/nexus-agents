"""断线语义的端到端验证：真起一个 uvicorn，真把连接掐掉。

为什么不能只测 ``_frames``
--------------------------
取消的触发点在 Starlette 的 ``StreamingResponse`` 里，而它是**服务器驱动**的：

* Starlette 1.10 按 ``scope["asgi"]["spec_version"]`` 分两条路径 ——
  ``>= 2.4`` 靠 ``send`` 抛 ``OSError``，``< 2.4`` 靠监听 ``http.disconnect``
  再取消发送任务。uvicorn 0.54 报的是 ``2.3``，走后者。
* ``httpx.ASGITransport`` 根本不产生 ``http.disconnect`` 消息（它还先把整个
  body 收完才返回），所以进程内测试只能验证「生成器被 aclose 之后会怎样」，
  验证不了「客户端掐线之后服务器会不会去 aclose 它」。

这一层如果坏了，症状是最贵的那种：用户关掉页面，服务端还在替他跑模型、
扣 token、占着会话锁。所以它值得一个真服务器的测试。

读流一律用带超时的 ``__anext__``：CI 里最糟的不是失败，是挂住。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest

from conftest import add
from nexus_agents import ReActAgent, Usage, assistant_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.service import ServiceSettings, create_app

uvicorn = pytest.importorskip("uvicorn", reason="断线语义只能由真 ASGI 服务器触发")

READ_TIMEOUT = 5.0


class GatedProvider:
    """把一轮请求冻在模型调用上，直到测试放行或被取消。"""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.calls = 0

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.calls += 1
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return LLMResponse(
            message=assistant_message("late answer"),
            usage=Usage(prompt_tokens=1, completion_tokens=1),
        )


@asynccontextmanager
async def _serve(app: object) -> AsyncIterator[str]:
    """在同一个事件循环里起 uvicorn，返回可访问的 base_url。"""
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning")
    server = uvicorn.Server(config)
    serving = asyncio.create_task(server.serve())
    try:
        for _ in range(500):  # 最多等 5 秒
            if server.started:
                break
            await asyncio.sleep(0.01)
        else:  # pragma: no cover - 起不来是环境问题，不该伪装成断言失败
            pytest.fail("uvicorn did not start")

        port = server.servers[0].sockets[0].getsockname()[1]
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        await asyncio.wait_for(serving, timeout=10)


async def _read_lines_until(resp: httpx.Response, marker: str) -> list[str]:
    """读到 `event: <marker>` 为止 —— 每次都带超时，不让 bug 变成挂起。"""
    seen: list[str] = []
    lines = resp.aiter_lines()
    while True:
        line = await asyncio.wait_for(lines.__anext__(), timeout=READ_TIMEOUT)
        if line.startswith("event:"):
            seen.append(line)
            if line.strip() == f"event: {marker}":
                return seen


async def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


@pytest.mark.asyncio
async def test_disconnect_over_real_http_cancels_the_run() -> None:
    provider = GatedProvider()
    app = create_app(
        ReActAgent(provider, [add]),
        settings=ServiceSettings(heartbeat_seconds=0.05),
    )

    async with _serve(app) as base_url, httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        session_id = (await client.post("/sessions", json={})).json()["session_id"]

        async with client.stream(
            "POST", f"/sessions/{session_id}/messages/stream", json={"text": "半路挂断"}
        ) as resp:
            assert resp.status_code == 200
            # 模型正在等的那一刻 == 用户关掉页面的那一秒
            await _read_lines_until(resp, "llm.call")

        assert provider.started.is_set()
        assert await _wait_until(lambda: provider.cancelled), "客户端断开后推理没有被取消"

        session = app.state.session_store.get(session_id)
        assert session is not None
        assert await _wait_until(lambda: not session.busy)
        assert session.turns == 0  # 半途而废的问答不进历史
        assert session.memory.recall("半路挂断") == ()


@pytest.mark.asyncio
async def test_session_is_usable_again_right_after_a_disconnect() -> None:
    """断线之后锁必须已经真的放手 —— 否则用户刷新页面就是 409。"""
    provider = GatedProvider()
    app = create_app(
        ReActAgent(provider, [add]),
        settings=ServiceSettings(heartbeat_seconds=0.05),
    )

    async with _serve(app) as base_url, httpx.AsyncClient(base_url=base_url, timeout=10) as client:
        session_id = (await client.post("/sessions", json={})).json()["session_id"]

        async with client.stream(
            "POST", f"/sessions/{session_id}/messages/stream", json={"text": "第一次"}
        ) as resp:
            await _read_lines_until(resp, "llm.call")

        assert await _wait_until(lambda: provider.cancelled)

        provider.release.set()  # 第二次请求放行
        second = await client.post(f"/sessions/{session_id}/messages", json={"text": "重发"})
        assert second.status_code == 200, second.text
        assert second.json()["answer"] == "late answer"

        session = app.state.session_store.get(session_id)
        assert session is not None
        assert session.turns == 1
