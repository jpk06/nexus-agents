"""HTTP service demo — runs offline, no API key, no network, no uvicorn.

为什么用 ``httpx.ASGITransport`` 而不是真起一个服务
----------------------------------------------
ASGI 应用本质是「一个接收 scope 的协程」。``ASGITransport`` 直接在进程内
调用它，不走 TCP。好处是示例零端口、零后台进程、CI 里也能跑；坏处是它
会把整个响应体收完才返回，所以「逐帧到达」的观感得靠端点自己保证 ——
下面第 3 段正是用时间戳证明这一点的。
"""

from __future__ import annotations

import asyncio
import time

import httpx

from nexus_agents import ReActAgent, tool
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.service import ServiceSettings, create_app
from nexus_agents.types import Usage, assistant_message


@tool
def stock(keyword: str) -> str:
    """Look up how many of something are in stock.

    Args:
        keyword: what to look for
    """
    return f"{keyword}: 12 件在库"


class SlowInventoryModel:
    """Pretends the model needs 120 ms — 用来演示心跳与 409。"""

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        await asyncio.sleep(0.12)
        return LLMResponse(
            message=assistant_message("茶杯还有 12 件"),
            usage=Usage(prompt_tokens=4, completion_tokens=6),
        )


def parse(body: str) -> list[tuple[str, str]]:
    """Tiny SSE parser: 客户端那边要写的就是这十几行。"""
    frames: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        event, data = "", ""
        for line in block.split("\n"):
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data = line[6:]
        if event:
            frames.append((event, data))
    return frames


async def main() -> None:
    provider = ScriptedProvider([tool_call_message("stock", {"keyword": "茶杯"}), "12 件"])
    app = create_app(
        ReActAgent(provider, [stock], name="warehouse"),
        settings=ServiceSettings(heartbeat_seconds=0.05),
    )
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://service") as client:
        # ---------------------------------------------------------- 1. 探针
        print("=== 1. healthz ===")
        print((await client.get("/healthz")).json())
        print()

        # ---------------------------------------------------------- 2. 建会话
        print("=== 2. POST /sessions ===")
        created = (await client.post("/sessions", json={"system": "你是仓库管理员"})).json()
        sid = created["session_id"]
        print({"session_id": sid, **{k: v for k, v in created.items() if k != "session_id"}})
        print()

        # ---------------------------------------------------------- 3. 一轮问答
        print("=== 3. POST /sessions/{id}/messages （JSON：审计路径）===")
        started = time.perf_counter()
        response = await client.post(f"/sessions/{sid}/messages", json={"text": "茶杯还有多少"})
        payload = response.json()
        elapsed = (time.perf_counter() - started) * 1000
        print(f"answer={payload['answer']!r} steps={payload['steps']} in {elapsed:.0f} ms")
        print(f"roles sent to the model: {[m['role'] for m in payload['messages']]}")
        print()

        # ---------------------------------------------------------- 4. SSE
        print("=== 4. POST .../messages/stream （SSE：聊天前端路径）===")
        started = time.perf_counter()
        async with client.stream(
            "POST", f"/sessions/{sid}/messages/stream", json={"text": "那上周呢"}
        ) as response:
            print(f"content-type: {response.headers['content-type']}")
            print(f"x-accel-buffering: {response.headers.get('x-accel-buffering')}")
            body = (await response.aread()).decode()
        for event, data in parse(body):
            stamp = f"+{(time.perf_counter() - started) * 1000:6.0f} ms"
            print(f"{stamp}  {event:<14} {data[:90]}")
        print()

        # ---------------------------------------------------------- 5. 409
        print("=== 5. same session, two turns at once -> 409 ===")
        slow_app = create_app(ReActAgent(SlowInventoryModel(), [stock]))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=slow_app), base_url="http://service"
        ) as slow:
            slow_sid = (await slow.post("/sessions", json={})).json()["session_id"]
            running = asyncio.create_task(
                slow.post(f"/sessions/{slow_sid}/messages", json={"text": "慢问题"})
            )
            await asyncio.sleep(0.02)  # 让第一轮先拿到锁
            second = await slow.post(f"/sessions/{slow_sid}/messages", json={"text": "插队"})
            print(f"second turn while the first is running: {second.status_code}")
            print(f"  detail: {second.json()['detail']}")
            first = await running
            print(f"first turn finished fine: {first.status_code}")
        print()

        # ---------------------------------------------------------- 6. 会话视图
        print("=== 6. GET /sessions/{id} （这轮聊了什么）===")
        view = (await client.get(f"/sessions/{sid}")).json()
        print(f"turns={view['turns']} messages={len(view['messages'])}")
        for message in view["messages"]:
            print(f"  {message['role']:<9} {message['content'][:40]}")

    print("\nall calls ran in-process — no server, no API key, no network")


if __name__ == "__main__":
    asyncio.run(main())
