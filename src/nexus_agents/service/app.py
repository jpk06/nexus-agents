"""HTTP layer: a session-scoped chat API with a live event stream.

为什么是 create_app(agent) 而不是模块级 ``app = FastAPI()``
----------------------------------------------------------
模块级 app 会在 import 时就固定住依赖：provider、工具集、会话存储全部变成
全局单例，测试只能靠 monkeypatch 去偷换，部署时也没法按环境换实现。
工厂模式把「装配」和「定义」分开：路由定义是纯的，谁都可以用任意 agent
装出一个 app —— 测试里装一个 ``ScriptedProvider`` 的 app，线上装一个真
模型的 app，路由代码一行不用改。

为什么 SSE 用 POST
------------------
浏览器原生的 ``EventSource`` 只能发 GET，不能带请求体，也不能加鉴权头。
把用户的问题塞进 URL 查询串既难看又有长度上限。所以这里用 POST +
``text/event-stream``，客户端用 ``fetch`` 读流（curl 也行）。代价是浏览器
端要自己解析帧格式，收益是请求体、鉴权头、幂等键全都能正常用。

为什么想清楚了「错误该在哪一层暴露」
------------------------------------
一旦开始流式响应，HTTP 状态码在第一个字节发出时就已经定死了。这时候
上游模型挂了，你**没法**把 200 改成 502 —— 只能在流里补一个 ``error``
事件。所以两条路径的错误处理不同：

* ``POST /messages``（JSON）：状态码有意义，映射成 502 / 500；
* ``POST /messages/stream``（SSE）：状态码早已发出，错误进 ``error`` 帧。

不显式区分，就会出现「客户端收到 200 + 半截答案，还以为成功了」。
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import secrets
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Annotated, Any, cast

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .. import __version__
from ..agent import AgentResult, ReActAgent
from ..errors import NexusError, ProviderError
from ..memory import ConversationMemory
from ..tracing import TraceEvent
from .sessions import InMemorySessionStore, Session, SessionStore
from .sse import SSE_HEADERS, SSE_MEDIA_TYPE, encode_comment, encode_event

__all__ = ["ServiceSettings", "create_app"]

logger = logging.getLogger("nexus_agents.service")


# ================================================================ 配置


@dataclass(frozen=True)
class ServiceSettings:
    """Knobs that belong to the service, not to the agent.

    分开的理由很实际：``AgentConfig`` 描述「一次运行怎么跑」，这里描述
    「一个服务怎么被人用」—— 输入上限、会话容量、心跳间隔。两者的变更
    节奏完全不同，混在一起会让 agent 越来越像框架。
    """

    #: 单条用户输入的字数上限。没有上限的接口就是一台内存放大器。
    max_input_chars: int = 8000
    #: 同时在线的会话数上限（超出按 LRU 淘汰）
    max_sessions: int = 1000
    #: 会话空闲多久后失效（秒）
    session_ttl_seconds: float = 3600.0
    #: SSE 心跳间隔（秒）。必须小于最常见的中间设备空闲超时（通常 60s）
    heartbeat_seconds: float = 15.0
    #: 每次请求带上的逐字历史条数上限。记忆负责「记得住」，历史负责「接得上」，
    #: 两者都要有；但历史不能无限长，否则提示词会随会话线性增长。
    history_messages: int = 40
    #: 可选共享密钥。设置后所有 /sessions 请求都要带 ``X-API-Key``。
    #: 这是演示级的闸门，不是身份认证 —— 生产环境请换成真正的鉴权。
    api_key: str | None = None

    def __post_init__(self) -> None:
        if self.max_input_chars < 1:
            raise ValueError("max_input_chars must be >= 1")
        if self.max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        if self.session_ttl_seconds <= 0:
            raise ValueError("session_ttl_seconds must be > 0")
        if self.heartbeat_seconds <= 0:
            raise ValueError("heartbeat_seconds must be > 0")
        if self.history_messages < 0:
            raise ValueError("history_messages must be >= 0")


# ================================================================ 请求/响应模型


class CreateSessionRequest(BaseModel):
    """建会话时唯一允许的输入。

    ``extra="forbid"`` 是刻意的：未知字段直接 422，而不是被静默忽略。
    静默忽略会让「我明明传了 model 参数怎么没生效」变成一次线上排查。
    """

    model_config = ConfigDict(extra="forbid")

    #: 会话级 system prompt，覆盖 agent 自身的默认配置
    system: str | None = Field(default=None, max_length=4000)

    @field_validator("system")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class CreateSessionResponse(BaseModel):
    session_id: str
    created_at: float


class SendMessageRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("text must not be blank")
        return value


class SessionView(BaseModel):
    """Audit view of a session: 说了什么、说了几轮。"""

    session_id: str
    created_at: float
    system: str | None
    turns: int
    messages: list[dict[str, Any]]


# ================================================================ 依赖


def get_settings(request: Request) -> ServiceSettings:
    return cast(ServiceSettings, request.app.state.settings)


def get_store(request: Request) -> SessionStore:
    return cast(SessionStore, request.app.state.session_store)


def get_agent(request: Request) -> ReActAgent:
    return cast(ReActAgent, request.app.state.agent)


async def require_api_key(
    request: Request,
    settings: ServiceSettings = Depends(get_settings),
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    """Opt-in shared secret. 比较用 ``compare_digest``，避免时序侧信道。

    时序攻击在演示项目里当然不现实，但「用 ``==`` 比密钥」是个会跟着代码
    一路复制到生产环境的坏习惯，成本只有一行。
    """
    expected = settings.api_key
    if expected is None:
        return
    if x_api_key is None or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=401,
            detail="invalid or missing X-API-Key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


async def require_session(
    session_id: str,
    store: SessionStore = Depends(get_store),
) -> Session:
    session = store.get(session_id)
    if session is None:
        # 404 而不是 410：调用方不需要区分「从没存在过」和「过期了」，
        # 两种情况的处置完全一样 —— 重新建一个。
        raise HTTPException(status_code=404, detail="session not found or expired")
    return session


# ================================================================ 路由


ops_router = APIRouter(tags=["ops"])
sessions_router = APIRouter(
    prefix="/sessions", tags=["sessions"], dependencies=[Depends(require_api_key)]
)


@ops_router.get("/healthz")
async def healthz(
    agent: ReActAgent = Depends(get_agent),
    store: SessionStore = Depends(get_store),
) -> dict[str, Any]:
    """存活探针。刻意不碰模型、不碰存储 —— 探针自己不能成为故障点。"""
    return {
        "status": "ok",
        "version": __version__,
        "agent": agent.name,
        "tools": list(agent.registry.names),
        "sessions": store.count(),
    }


@sessions_router.post("", status_code=201, response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest | None = None,
    store: SessionStore = Depends(get_store),
) -> CreateSessionResponse:
    session = store.create(
        system=body.system if body is not None else None,
        memory=ConversationMemory(),
    )
    return CreateSessionResponse(session_id=session.id, created_at=session.created_at)


@sessions_router.get("/{session_id}", response_model=SessionView)
async def read_session(session: Session = Depends(require_session)) -> SessionView:
    return SessionView(
        session_id=session.id,
        created_at=session.created_at,
        system=session.system,
        turns=session.turns,
        messages=[m.model_dump(mode="json") for m in session.messages],
    )


@sessions_router.delete("/{session_id}", status_code=204, response_class=Response)
async def delete_session(
    session: Session = Depends(require_session),
    store: SessionStore = Depends(get_store),
) -> Response:
    store.delete(session.id)
    return Response(status_code=204)


@sessions_router.post("/{session_id}/messages")
async def send_message(
    body: SendMessageRequest,
    session: Session = Depends(require_session),
    agent: ReActAgent = Depends(get_agent),
    settings: ServiceSettings = Depends(get_settings),
) -> dict[str, Any]:
    """Run one turn and return the complete result.

    这是「审计 / 调试」路径：返回 ``AgentResult.to_dict()``（含 messages 与
    trace）。聊天前端请用 SSE 版本 —— 它不会让你盯着空白屏幕等 30 秒。
    """
    _check_length(body.text, settings)
    async with _exclusive(session):
        try:
            result = await agent.arun(
                body.text,
                system=session.system,
                memory=session.memory,
                history=session.recent_history(settings.history_messages),
            )
        except ProviderError as exc:
            # 502：问题出在上游模型，不是这个服务的 bug
            logger.warning("provider failed for session %s: %s", session.id, exc)
            raise HTTPException(status_code=502, detail="upstream model failed") from exc
        except NexusError as exc:
            logger.exception("agent run failed for session %s", session.id)
            raise HTTPException(status_code=500, detail="agent run failed") from exc
    session.record(result.messages)
    return result.to_dict()


@sessions_router.post("/{session_id}/messages/stream")
async def stream_message(
    body: SendMessageRequest,
    session: Session = Depends(require_session),
    agent: ReActAgent = Depends(get_agent),
    settings: ServiceSettings = Depends(get_settings),
) -> StreamingResponse:
    """Same turn, streamed as Server-Sent Events.

    帧序列：``agent.start`` → 若干 ``step.start`` / ``llm.call`` /
    ``llm.response`` / ``tool.call`` / ``tool.result`` → ``agent.result``。
    出错时最后补一个 ``error`` 帧 —— 状态码已经发出去了，无法再改。
    """
    _check_length(body.text, settings)
    # 先造响应，再抢锁：StreamingResponse 的构造不会启动生成器，所以即使
    # 这里 409 返回，也不会留下一个「拿着锁却没人消费」的生成器。
    # 反过来写就必须用 try/except 手工还锁 —— 少一条清理路径，少一类漏锁。
    response = StreamingResponse(
        _frames(agent, session, body.text, settings),
        media_type=SSE_MEDIA_TYPE,
        headers=SSE_HEADERS,
    )
    await _acquire_or_409(session)
    return response


# ================================================================ 内部实现


def _check_length(text: str, settings: ServiceSettings) -> None:
    if len(text) > settings.max_input_chars:
        # 413 而不是 422：请求本身合法，只是太大了
        raise HTTPException(
            status_code=413,
            detail=f"text too long: {len(text)} > {settings.max_input_chars} characters",
        )


@contextlib.asynccontextmanager
async def _exclusive(session: Session) -> AsyncIterator[Session]:
    """串行化同一会话的两次运行，忙则直接 409。

    为什么拒绝而不是排队
    --------------------
    排队看起来更「友好」，但队列没有上限时就变成了内存放大器：一个人狂点
    发送，服务就替他攒下几百个待跑的模型调用。409 让客户端立刻知道
    「你上一个还没回」，这是更诚实的背压。
    """
    await _acquire_or_409(session)
    try:
        yield session
    finally:
        session.lock.release()


async def _acquire_or_409(session: Session) -> None:
    """``locked()`` 与 ``acquire()`` 之间没有 await，所以检查不会失效。

    这是单线程事件循环里少见的「真的可以这么做」的场景 —— 一旦中间插进
    任何一个 await，这里就必须换成 ``lock.acquire()`` + 超时。
    """
    if session.busy:
        raise HTTPException(
            status_code=409, detail="session is busy with another turn; retry later"
        )
    await session.lock.acquire()


def _result_payload(result: AgentResult) -> dict[str, Any]:
    """SSE 上的结果帧：只发客户端要用的，不重复发 trace。

    trace 已经逐条流过了，messages 里也有整段对话 —— 再塞一遍只会让最后
    一帧变成几十 KB，而客户端一个字段都用不上。
    """
    return {
        "answer": result.answer,
        "stopped_reason": result.stopped_reason.value,
        "succeeded": result.succeeded,
        "steps": result.steps,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
            "total_tokens": result.usage.total_tokens,
        },
        "message_count": len(result.messages),
    }


def _event_payload(event: TraceEvent) -> dict[str, Any]:
    return {"step": event.step, "timestamp": event.timestamp, **event.payload}


async def _frames(
    agent: ReActAgent,
    session: Session,
    text: str,
    settings: ServiceSettings,
) -> AsyncIterator[str]:
    """Turn ``agent.astream`` into SSE frames, with heartbeats and clean cancel.

    两个关键决定
    ------------
    1. **心跳不能取消正在等的那次拉取**。``asyncio.wait_for(anext(...))`` 超时
       会把 ``anext`` 取消掉，而这等于往 agent 的生成器里扔一个
       ``CancelledError`` —— 心跳一跳，整轮推理就死了。所以这里用
       ``asyncio.wait`` 只等「结果到没到」，超时就发一行注释，**同一个
       task 继续挂着**。任何时刻只有一个未完成的拉取，背压依然真实。
    2. **客户端断开等于取消**。``finally`` 里取消挂起的拉取并 ``aclose()``
       生成器；agent 的契约是「中途放弃就不写记忆」，半途而废的问答不该
       变成这个会话的长期记忆。
    """
    iterator = agent.astream(
        text,
        system=session.system,
        memory=session.memory,
        history=session.recent_history(settings.history_messages),
    )
    pending: asyncio.Task[TraceEvent | AgentResult] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            done, _ = await asyncio.wait({pending}, timeout=settings.heartbeat_seconds)
            if not done:
                yield encode_comment("keep-alive")
                continue
            task, pending = pending, None
            try:
                item = task.result()
            except StopAsyncIteration:
                break
            # 注意 CancelledError 不会进这里：3.8 起它是 BaseException。
            # 取消就该一路往上抛，绝不能被「转成 error 帧」吞掉。
            except Exception as exc:  # 流一旦开始，错误只能就地转成事件
                logger.exception("agent stream failed for session %s", session.id)
                yield encode_event("error", _error_payload(exc))
                return

            if isinstance(item, AgentResult):
                session.record(item.messages)
                yield encode_event("agent.result", _result_payload(item))
            else:
                yield encode_event(item.type, _event_payload(item))
    finally:
        if pending is not None and not pending.done():
            # 取消挂起的拉取 = 让 agent 在下一个 await 点停下。
            # 光``cancel()``不够 —— 取消是「登记」而不是「立即生效」，
            # 不等它落地就释放锁，下一轮可能和还没停的旧推理并行跑。
            pending.cancel()
            # 已经在收尾，只关心它停没停：错误类型此时无关紧要
            with contextlib.suppress(BaseException):
                await pending
        with contextlib.suppress(RuntimeError, asyncio.CancelledError):
            await iterator.aclose()
        if session.lock.locked():
            session.lock.release()


def _error_payload(exc: BaseException) -> Mapping[str, Any]:
    """错误帧的 payload。

    只对自家异常暴露 message：模型的 transport 错误里可能有内部 URL，
    陌生异常一律折成一句通用描述，细节进服务端日志。
    """
    if isinstance(exc, NexusError):
        return {"type": type(exc).__name__, "message": str(exc)}
    return {"type": "InternalError", "message": "internal error while running the agent"}


# ================================================================ 工厂


def create_app(
    agent: ReActAgent,
    *,
    settings: ServiceSettings | None = None,
    store: SessionStore | None = None,
) -> FastAPI:
    """Build the ASGI app around one agent.

    agent 实例可以同时服务多个会话：一次 run 不往 agent 上写任何状态
    （见 ``ReActAgent.run`` 的说明），会话状态全在 ``Session`` 里。
    """
    resolved = settings or ServiceSettings()
    app = FastAPI(
        title="nexus-agents service",
        version=__version__,
        summary="Session-scoped chat API with a live SSE event stream.",
    )
    app.state.agent = agent
    app.state.settings = resolved
    app.state.session_store = store or InMemorySessionStore(
        max_sessions=resolved.max_sessions,
        ttl_seconds=resolved.session_ttl_seconds,
    )
    app.include_router(ops_router)
    app.include_router(sessions_router)
    return app
