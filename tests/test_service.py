"""HTTP layer tests — offline, and adversarial about the parts that usually rot.

这一层最容易「demo 里能跑、线上全是坑」。四类性质值得逐条钉死：

1. **会话隔离与连续性**：两个会话不能互相污染；同一个会话的第二轮必须
   真的看得见第一轮（靠逐字 history，不是靠运气好被 BM25 召回）。
2. **并发**：同一会话同时两轮必须被拒绝（409），而不是把记忆写乱。
3. **取消**：客户端半路断开 → 锁必须释放、这一轮绝不能进记忆。
4. **错误分层**：JSON 路径靠状态码，SSE 路径只能靠 in-band 的 ``error`` 帧。

心跳那条尤其反直觉：验证方式不是「有 keep-alive 字样」，而是「心跳之后
整轮仍然跑完了」—— 一个把 ``anext`` 取消掉的实现照样能发出心跳，然后
把推理杀在半路。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from conftest import add
from nexus_agents import AgentError, ReActAgent, ToolCall, ToolResult, assistant_message
from nexus_agents.errors import ProviderTransportError
from nexus_agents.memory import ConversationMemory
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.service import (
    InMemorySessionStore,
    ServiceSettings,
    create_app,
    encode_comment,
    encode_event,
)
from nexus_agents.service.app import _frames  # 断开连接只能绕过 ASGI 直接测
from nexus_agents.service.sessions import Session
from nexus_agents.tracing import EventType
from nexus_agents.types import Usage, tool_result_message, user_message

# ================================================================ 测试替身


class GatedProvider:
    """Finishes only when the test opens the gate — 用来把一轮请求冻在半路上。"""

    def __init__(self, reply: str = "gated answer") -> None:
        self.reply = reply
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        self.started.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        return LLMResponse(
            message=assistant_message(self.reply),
            usage=Usage(prompt_tokens=2, completion_tokens=2),
        )


class SlowProvider:
    """Real delay, so the heartbeat path is exercised for real."""

    def __init__(self, delay: float, reply: str = "slow answer") -> None:
        self.delay = delay
        self.reply = reply

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        await asyncio.sleep(self.delay)
        return LLMResponse(
            message=assistant_message(self.reply),
            usage=Usage(prompt_tokens=1, completion_tokens=1),
        )


class ExplodingProvider:
    def __init__(self, message: str = "upstream refused") -> None:
        self.message = message

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        raise ProviderTransportError(self.message)


class LeakyProvider:
    """Raises something that is *not* ours — 它的 message 绝不能出现在响应里。"""

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        raise RuntimeError("postgres://user:hunter2@internal/db")


# ================================================================ 工具函数


def parse_sse(body: str) -> list[dict[str, Any]]:
    """Parse our own frames back into ``{event, data, id}`` dicts.

    故意不复用服务端的编码器：拿同一个函数「编码再解码」验证不出任何东西。
    """
    frames: list[dict[str, Any]] = []
    for block in body.split("\n\n"):
        if not block.strip() or block.lstrip().startswith(":"):
            continue
        frame: dict[str, Any] = {"data_lines": []}
        for line in block.split("\n"):
            field, _, value = line.partition(": ")
            if field == "event":
                frame["event"] = value
            elif field == "data":
                frame["data_lines"].append(value)
            elif field == "id":
                frame["id"] = value
        raw = "\n".join(frame.pop("data_lines"))
        frame["data"] = json.loads(raw) if raw else None
        frames.append(frame)
    return frames


def build_app(provider: Any, **settings: Any):
    agent = ReActAgent(provider, [add])
    app = create_app(agent, settings=ServiceSettings(**settings))
    return app


def client_for(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    )


# ================================================================ SSE 编码（纯函数）


def test_encode_event_defaults_to_json_without_escaping_unicode() -> None:
    frame = encode_event("agent.result", {"answer": "五", "steps": 1})
    assert frame == 'event: agent.result\ndata: {"answer":"五","steps":1}\n\n'


def test_encode_event_splits_multiline_payload_into_separate_data_lines() -> None:
    frame = encode_event("note", "first\nsecond")
    assert frame == "event: note\ndata: first\ndata: second\n\n"


def test_encode_event_keeps_empty_payload_as_one_empty_data_line() -> None:
    # 没有 data 行的帧在浏览器端不会触发 message 事件 —— 空串也必须发
    assert encode_event("ping", "") == "event: ping\ndata: \n\n"


def test_encode_event_rejects_newline_in_name() -> None:
    with pytest.raises(ValueError, match="single line"):
        encode_event("bad\nevent", {})


def test_encode_comment_uses_sse_comment_syntax() -> None:
    assert encode_comment("keep-alive") == ": keep-alive\n\n"


# ================================================================ 会话存储


def test_store_expires_sessions_using_monotonic_clock() -> None:
    now = [1000.0]
    store = InMemorySessionStore(ttl_seconds=10, clock=lambda: now[0])
    session = store.create()

    now[0] += 9
    assert store.get(session.id) is session  # 命中也算「用过」，续命
    now[0] += 12
    assert store.get(session.id) is None
    assert store.count() == 0


def test_store_evicts_least_recently_used_when_full() -> None:
    store = InMemorySessionStore(max_sessions=2, ttl_seconds=1000)
    first = store.create()
    second = store.create()
    store.get(first.id)  # 让 first 变成最近使用
    third = store.create()

    assert store.get(second.id) is None
    assert store.get(first.id) is first
    assert store.get(third.id) is third
    assert store.evicted == 1


def test_store_delete_reports_whether_anything_was_there() -> None:
    store = InMemorySessionStore()
    session = store.create()
    assert store.delete(session.id) is True
    assert store.delete(session.id) is False
    assert store.count() == 0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"max_sessions": 0}, "max_sessions"), ({"ttl_seconds": 0}, "ttl_seconds")],
)
def test_store_rejects_nonsense_limits(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        InMemorySessionStore(**kwargs)


def test_recent_history_never_starts_mid_tool_call() -> None:
    """窗口切在 assistant(tool_calls) 和 tool 结果之间 = 一个模型必然拒收的请求。"""
    session = Session(id="s", memory=ConversationMemory(), created_at=0.0)
    session.messages = [
        user_message("q1"),
        assistant_message(tool_calls=(ToolCall(id="c1", name="add", arguments={}),)),
        tool_result_message(ToolResult(tool_call_id="c1", name="add", output="3")),
        assistant_message("answer 1"),
        user_message("q2"),
        assistant_message("answer 2"),
    ]

    # 只要最后 3 条会从 assistant(tool_calls) 开始 —— 那就必须跳到 q2
    assert [m.content for m in session.recent_history(3)] == ["q2", "answer 2"]
    assert [m.content for m in session.recent_history(40)] == [
        "q1",
        "",
        "3",
        "answer 1",
        "q2",
        "answer 2",
    ]
    assert session.recent_history(0) == []


def test_recent_history_returns_empty_when_window_holds_no_user_message() -> None:
    session = Session(id="s", memory=ConversationMemory(), created_at=0.0)
    session.messages = [assistant_message("only assistant")]
    assert session.recent_history(5) == []


# ================================================================ 基础路由


@pytest.mark.asyncio
async def test_healthz_reports_version_and_tool_names() -> None:
    async with client_for(build_app(ScriptedProvider(["hi"]))) as client:
        body = (await client.get("/healthz")).json()

    assert body["status"] == "ok"
    assert body["tools"] == ["add"]
    assert body["sessions"] == 0


@pytest.mark.asyncio
async def test_create_session_returns_identifier_and_timestamp() -> None:
    async with client_for(build_app(ScriptedProvider(["hi"]))) as client:
        response = await client.post("/sessions", json={})

    assert response.status_code == 201
    payload = response.json()
    assert len(payload["session_id"]) == 32
    assert payload["created_at"] > 0


@pytest.mark.asyncio
async def test_create_session_accepts_optional_system_prompt() -> None:
    app = build_app(ScriptedProvider(["hi"]))
    async with client_for(app) as client:
        response = await client.post("/sessions", json={"system": "  你是库管  "})
        view = await client.get(f"/sessions/{response.json()['session_id']}")

    assert view.json()["system"] == "你是库管"


@pytest.mark.asyncio
async def test_unknown_session_is_404_on_every_route() -> None:
    async with client_for(build_app(ScriptedProvider(["hi"]))) as client:
        missing = "/sessions/does-not-exist"
        statuses = [
            (await client.get(missing)).status_code,
            (await client.post(f"{missing}/messages", json={"text": "hi"})).status_code,
            (await client.post(f"{missing}/messages/stream", json={"text": "hi"})).status_code,
            (await client.delete(missing)).status_code,
        ]

    assert statuses == [404, 404, 404, 404]


@pytest.mark.asyncio
async def test_delete_session_removes_it() -> None:
    store = InMemorySessionStore()
    app = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]), store=store)
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        assert (await client.delete(f"/sessions/{sid}")).status_code == 204
        assert store.count() == 0


# ================================================================ 一轮问答（JSON）


@pytest.mark.asyncio
async def test_send_message_returns_full_audit_trail() -> None:
    provider = ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"])
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages", json={"text": "2+3?"})

    payload = response.json()
    assert response.status_code == 200
    assert payload["answer"] == "5"
    assert payload["stopped_reason"] == "final"
    assert payload["steps"] == 2
    # JSON 路径是审计路径：messages 与 trace 都要在
    assert [m["role"] for m in payload["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert payload["trace"][0]["type"] == EventType.AGENT_START


@pytest.mark.asyncio
async def test_second_turn_really_sees_the_first_one() -> None:
    """关键词完全不同也要接得上 —— 这靠的是逐字 history，不是 BM25 召回。"""
    provider = ScriptedProvider(["新箱子的容量是 12 升", "因为它有 12 升"])
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        await client.post(f"/sessions/{sid}/messages", json={"text": "那个箱子多大"})
        await client.post(f"/sessions/{sid}/messages", json={"text": "为什么"})

    sent = [m.content for m in provider.last_request.messages]
    assert "那个箱子多大" in sent
    assert "新箱子的容量是 12 升" in sent
    assert sent[-1] == "为什么"  # 新问题永远在最后


@pytest.mark.asyncio
async def test_sessions_do_not_share_history() -> None:
    provider = ScriptedProvider(["答案 A", "答案 B"])
    async with client_for(build_app(provider)) as client:
        first = (await client.post("/sessions", json={})).json()["session_id"]
        second = (await client.post("/sessions", json={})).json()["session_id"]
        await client.post(f"/sessions/{first}/messages", json={"text": "只在 A 里说的话"})
        await client.post(f"/sessions/{second}/messages", json={"text": "B 的问题"})

    sent = [m.content for m in provider.last_request.messages]
    assert "只在 A 里说的话" not in sent
    assert "答案 A" not in sent


@pytest.mark.asyncio
async def test_session_view_reports_turns_and_skips_injected_system_messages() -> None:
    provider = ScriptedProvider(["好的"])
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={"system": "你是库管"})).json()["session_id"]
        await client.post(f"/sessions/{sid}/messages", json={"text": "查库存"})
        view = (await client.get(f"/sessions/{sid}")).json()

    assert view["turns"] == 1
    assert [m["role"] for m in view["messages"]] == ["user", "assistant"]


# ================================================================ 输入校验


@pytest.mark.asyncio
async def test_blank_text_is_rejected_before_any_model_call() -> None:
    provider = ScriptedProvider(["never"])
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages", json={"text": "   "})

    assert response.status_code == 422
    assert provider.call_count == 0


@pytest.mark.asyncio
async def test_unknown_fields_are_rejected_instead_of_ignored() -> None:
    async with client_for(build_app(ScriptedProvider(["x"]))) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(
            f"/sessions/{sid}/messages", json={"text": "hi", "max_steps": 99}
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_oversized_input_is_413_not_422() -> None:
    app = build_app(ScriptedProvider(["x"]), max_input_chars=5)
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages", json={"text": "0123456789"})

    assert response.status_code == 413
    assert "10 > 5" in response.json()["detail"]


def test_settings_validate_their_own_limits() -> None:
    with pytest.raises(ValueError, match="heartbeat_seconds"):
        ServiceSettings(heartbeat_seconds=0)
    with pytest.raises(ValueError, match="history_messages"):
        ServiceSettings(history_messages=-1)


# ================================================================ 鉴权


@pytest.mark.asyncio
async def test_api_key_is_enforced_only_when_configured() -> None:
    app = build_app(ScriptedProvider(["hi"]), api_key="s3cret")
    async with client_for(app) as client:
        assert (await client.post("/sessions", json={})).status_code == 401
        assert (
            await client.post("/sessions", json={}, headers={"X-API-Key": "wrong"})
        ).status_code == 401
        assert (
            await client.post("/sessions", json={}, headers={"X-API-Key": "s3cret"})
        ).status_code == 201
        # 探针不设防：健康检查被鉴权挡住，编排系统会以为服务挂了
        assert (await client.get("/healthz")).status_code == 200


# ================================================================ 并发


@pytest.mark.asyncio
async def test_concurrent_turn_on_same_session_is_409() -> None:
    """一轮还在跑，第二轮必须立刻被拒 —— 而不是排队把记忆写乱。

    ``httpx.ASGITransport`` 会把整个响应体收完才返回（不暴露 HTTP 的
    「先发头、后发体」），所以这里用后台 task 模拟「客户端仍在等流」，
    而不是依赖 ``client.stream`` 的惰性读取。
    """
    provider = GatedProvider()
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]

        streaming = asyncio.create_task(
            client.post(f"/sessions/{sid}/messages/stream", json={"text": "慢问题"})
        )
        await asyncio.wait_for(provider.started.wait(), timeout=2)

        second = await client.post(f"/sessions/{sid}/messages", json={"text": "插队"})
        assert second.status_code == 409

        provider.release.set()
        first = await asyncio.wait_for(streaming, timeout=2)
        assert first.status_code == 200

        third = await client.post(f"/sessions/{sid}/messages", json={"text": "下一轮"})
    assert third.status_code == 200


# ================================================================ SSE 流


@pytest.mark.asyncio
async def test_stream_emits_events_in_order_and_ends_with_result() -> None:
    provider = ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"])
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(
            f"/sessions/{sid}/messages/stream", json={"text": "2+3?"}
        )

    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    frames = parse_sse(response.text)
    events = [f["event"] for f in frames]

    assert events[0] == EventType.AGENT_START
    assert events[-1] == "agent.result"
    assert EventType.TOOL_CALL in events and EventType.TOOL_RESULT in events
    assert [f["data"]["answer"] for f in frames if f["event"] == "agent.result"] == ["5"]
    # 最后一帧不该把整段对话和 trace 再塞一遍
    assert "trace" not in frames[-1]["data"] and "messages" not in frames[-1]["data"]


@pytest.mark.asyncio
async def test_stream_records_the_turn_for_later_reads() -> None:
    async with client_for(build_app(ScriptedProvider(["好的"]))) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        await client.post(f"/sessions/{sid}/messages/stream", json={"text": "第一句"})
        view = (await client.get(f"/sessions/{sid}")).json()

    assert view["turns"] == 1
    assert view["messages"][0]["content"] == "第一句"


@pytest.mark.asyncio
async def test_heartbeat_keeps_the_run_alive_instead_of_killing_it() -> None:
    """心跳必须只发注释，不能取消正在等的那次拉取。"""
    provider = SlowProvider(delay=0.12)
    app = build_app(provider, heartbeat_seconds=0.02)
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages/stream", json={"text": "慢"})

    assert ": keep-alive" in response.text
    frames = parse_sse(response.text)
    assert frames[-1]["event"] == "agent.result"
    assert frames[-1]["data"]["answer"] == "slow answer"


@pytest.mark.asyncio
async def test_provider_failure_mid_stream_becomes_an_error_frame() -> None:
    app = build_app(ExplodingProvider("connection refused by model host"))
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages/stream", json={"text": "hi"})
        view = (await client.get(f"/sessions/{sid}")).json()

    # 头已经发出去了，状态码只能是 200 —— 错误信息必须在流里
    assert response.status_code == 200
    frames = parse_sse(response.text)
    assert frames[-1]["event"] == "error"
    assert frames[-1]["data"]["type"] == "ProviderTransportError"
    assert frames[-1]["data"]["message"] == "connection refused by model host"
    assert view["turns"] == 0  # 失败的一轮不算一轮


@pytest.mark.asyncio
async def test_foreign_exception_details_never_leak_to_the_client() -> None:
    app = build_app(LeakyProvider())
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages/stream", json={"text": "hi"})

    frames = parse_sse(response.text)
    assert frames[-1]["data"]["type"] == "InternalError"
    assert "hunter2" not in response.text


@pytest.mark.asyncio
async def test_client_disconnect_releases_the_lock_and_skips_memory() -> None:
    """真实的断开语义只能绕过 ASGI 直接验：关闭生成器 == 客户端挂断。"""
    provider = GatedProvider()
    agent = ReActAgent(provider, [add])
    session = Session(id="s", memory=ConversationMemory(), created_at=0.0)
    settings = ServiceSettings(heartbeat_seconds=0.05)

    frames = _frames(agent, session, "半路挂断", settings).__aiter__()
    assert "agent.start" in await frames.__anext__()
    # llm.call 在调用「发起时」就发出，所以拿到它时模型还在等 —— 这正是
    # 客户端会拔网线的那一秒
    assert "llm.call" in await frames.__anext__()
    # 再拉一次：拉取被模型卡住，超时后只发出一行心跳，task 仍然挂着
    assert (await frames.__anext__()).strip() == ": keep-alive"
    assert provider.started.is_set()  # 模型确实已经进入了这次调用

    await frames.aclose()

    assert provider.cancelled is True  # 推理真的被叫停了，不是继续在后台烧钱
    assert session.busy is False
    assert session.turns == 0
    assert session.memory.recall("半路挂断") == ()


@pytest.mark.asyncio
async def test_stream_releases_lock_even_when_the_run_raises() -> None:
    agent = ReActAgent(ExplodingProvider(), [add])
    session = Session(id="s", memory=ConversationMemory(), created_at=0.0)

    chunks = [chunk async for chunk in _frames(agent, session, "hi", ServiceSettings())]

    assert session.busy is False
    assert chunks[-1].startswith("event: error")


# ================================================================ JSON 路径的错误映射


@pytest.mark.asyncio
async def test_provider_failure_on_json_path_is_502() -> None:
    async with client_for(build_app(ExplodingProvider())) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages", json={"text": "hi"})

    assert response.status_code == 502
    assert response.json()["detail"] == "upstream model failed"


@pytest.mark.asyncio
async def test_json_path_failure_still_releases_the_lock() -> None:
    async with client_for(build_app(ExplodingProvider())) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        await client.post(f"/sessions/{sid}/messages", json={"text": "hi"})
        retry = await client.post(f"/sessions/{sid}/messages", json={"text": "hi"})

    assert retry.status_code == 502  # 不是 409 —— 锁没漏


# ================================================================ 收尾：把边缘分支也钉住


def test_encode_event_supports_an_explicit_id() -> None:
    frame = encode_event("step", {"n": 1}, event_id="7")
    assert frame.startswith("id: 7\nevent: step\n")
    with pytest.raises(ValueError, match="event id"):
        encode_event("step", {}, event_id="7\n8")


def test_encode_event_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        encode_event("", {})


def test_encode_comment_keeps_multiple_lines_separate() -> None:
    assert encode_comment("a\nb") == ": a\n: b\n\n"


def test_blank_system_prompt_normalises_to_none() -> None:
    @pytest.mark.asyncio
    async def _run() -> None:
        async with client_for(build_app(ScriptedProvider(["x"]))) as client:
            created = await client.post("/sessions", json={"system": "   "})
            view = await client.get(f"/sessions/{created.json()['session_id']}")
            explicit_null = await client.post("/sessions", json={"system": None})
            null_view = await client.get(f"/sessions/{explicit_null.json()['session_id']}")
        assert view.json()["system"] is None
        assert null_view.json()["system"] is None

    asyncio.run(_run())


@pytest.mark.asyncio
async def test_store_purges_expired_sessions_when_creating_a_new_one() -> None:
    now = [0.0]
    store = InMemorySessionStore(ttl_seconds=10, clock=lambda: now[0])
    stale = store.create()
    now[0] += 11
    store.create()
    assert store.get(stale.id) is None
    assert store.count() == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_input_chars": 0}, "max_input_chars"),
        ({"max_sessions": 0}, "max_sessions"),
        ({"session_ttl_seconds": 0}, "session_ttl_seconds"),
    ],
)
def test_settings_reject_nonsense_limits(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ServiceSettings(**kwargs)


class AgentBlowUpProvider:
    """Raises a NexusError that is *not* a provider error — 内部契约坏了。"""

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        raise AgentError("stream ended without a result")


@pytest.mark.asyncio
async def test_internal_agent_failure_is_500_not_502() -> None:
    async with client_for(build_app(AgentBlowUpProvider())) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        response = await client.post(f"/sessions/{sid}/messages", json={"text": "hi"})

    assert response.status_code == 500
    assert response.json()["detail"] == "agent run failed"
