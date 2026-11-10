"""Observability: request IDs and JSON logs that a machine can grep.

为什么日志要结构化
------------------
``logger.info("session %s done in %.0fms", sid, ms)`` 产出的那行人类可读文本，
对 grep 友好、对 ELK / Loki / CloudWatch 这类采集管道**不**友好：管道里没有
「哪一段是 session_id」的元数据，只能靠正则反解，一条格式改动就让全部历史
日志失去结构。JSON Lines（一行一个对象）是采集侧的事实标准：字段名即索引。

两个刻意的小决定
----------------
1. **Formatter 不引入任何第三方依赖**（structlog/python-json-logger 都很好，
   但核心仓库的哲学是零运行时依赖，日志不值得破例）。手写 ~40 行就能得到
   「一行一个对象 + extras 透传 + 异常折叠成字段」的 90% 价值。
2. **中间件写成纯 ASGI**，不用 Starlette 的 ``BaseHTTPMiddleware``。后者要把
   响应包一层再交给用户代码，历史上多次和流式响应（正是我们的 SSE）打架；
   纯 ASGI 只包 ``send`` 这一个函数，流式行为零改动。

请求 ID 的契约
--------------
* 客户端带了合法 ``X-Request-ID`` 就沿用（链路里多个服务能串成一条）；
* 非法（太长、含换行）或缺失就生成 ``secrets.token_hex(8)``；
* 原样回写响应头 —— 客户端报障时只要给这个 ID，就能从日志里捞出整条请求。
"""

from __future__ import annotations

import json
import logging
import secrets
import sys
import time
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = [
    "JsonFormatter",
    "RequestLoggingMiddleware",
    "configure_logging",
    "current_request_id",
]

#: 请求 ID 上限。够 ``token_hex(8)`` 的两倍还有余；超长的输入一律不信任，
#: 否则日志和响应头都成了外部输入的放大器。
_MAX_REQUEST_ID_LENGTH = 128

#: LogRecord 的标准属性集合。Formatter 只把**不**在这里的属性当作用户 extras
#: 透传进 JSON —— 这样 ``logger.info(..., extra={"request_id": rid})`` 的字段
#: 会自动出现在输出里，Formatter 不需要认识每一个字段名。
_RESERVED_ATTRS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "id", "levelname", "levelno", "lineno", "module", "msecs",
        "message", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "stack_info_text", "thread",
        "threadName", "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render each LogRecord as one line of JSON.

    输出形状（稳定契约，采集侧按字段名建索引）::

        {"ts": "...", "level": "info", "logger": "nexus_agents.service",
         "event": "http_request", "request_id": "ab12...", "status": 200}

    ``event`` 就是原来的 message（已格式化）。异常不渲染成多行 traceback 拼在
    message 里，而是折成 ``exception`` 字段 —— 一行日志仍然是一行。
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        # extras 透传：extra={...} 里的字段直接平铺进顶层。
        # 值不保证可序列化（可能是异常对象、dataclass），default=str 兜底。
        for key, value in record.__dict__.items():
            if key not in _RESERVED_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            # 只在发生异常时占一个字段；exc_text 是缓存，可能还没生成
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(
    level: str = "info",
    *,
    json_output: bool = False,
    logger_name: str = "nexus_agents",
    stream: Any = None,
) -> logging.Logger:
    """Attach exactly one output handler to the package logger.

    幂等：重复调用先摘掉**自己**上一次加的 handler（用 ``_nexus_owned`` 标记），
    不会碰用户/uvicorn 配置的任何其他 handler —— ``__main__`` 每次启动调它，
    测试里反复调它，都不会堆出重复行。

    ``stream`` 参数只为测试存在：生产路径永远写 stderr（stdout 在容器里有时
    被框架占用，stderr 是 12-factor 默认的日志通道）。
    """
    target = logging.getLogger(logger_name)
    for handler in list(target.handlers):
        if getattr(handler, "_nexus_owned", False):
            target.removeHandler(handler)
            handler.close()
    handler = logging.StreamHandler(stream if stream is not None else sys.stderr)
    handler.setFormatter(JsonFormatter() if json_output else logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s"
    ))
    handler._nexus_owned = True  # type: ignore[attr-defined]
    target.addHandler(handler)
    target.setLevel(level.upper())
    # 自带 handler 之后不再向 root 传播：否则 pytest/应用配置的 root handler
    # 会把同一行再打一遍。
    target.propagate = False
    return target


# ================================================================ 请求 ID


def _sanitize_request_id(raw: str | None) -> str:
    """信任但有验证：合法沿用，非法换新。

    两个检查各挡一类真实攻击：长度挡「写满日志/响应头」的放大，换行挡
    **日志注入**（把 ``\\n`` 塞进 ID，让一行日志变成两行伪造记录）。
    """
    if raw and len(raw) <= _MAX_REQUEST_ID_LENGTH and "\n" not in raw and "\r" not in raw:
        return raw
    return secrets.token_hex(8)


def current_request_id(scope: Scope) -> str | None:
    """Read the request ID the middleware stashed in the ASGI scope.

    路由处理器用它把业务日志（provider 失败、流中断）和访问日志串到同一个
    ID 上。中间件不在（比如单元测试直接调 handler）时返回 ``None``，调用方
    照常记日志，只是缺一个字段。
    """
    state = scope.get("state")
    if isinstance(state, MutableMapping):
        value = state.get("request_id")
        if isinstance(value, str):
            return value
    return None


# ================================================================ 访问日志中间件


class RequestLoggingMiddleware:
    """Pure-ASGI access log: one line per request, with request ID and duration.

    为什么跳过 ``/healthz``：编排器（Compose healthcheck、K8s liveness probe）
    会**每几秒**打一次探针。把它们也记进访问日志，等于让日志量与在线时长
    成正比、而与真实流量无关 —— 排障时还要先滤掉 99% 的探针噪音。
    """

    def __init__(self, app: ASGIApp, *, logger: logging.Logger | None = None) -> None:
        self.app = app
        self.logger = logger or logging.getLogger("nexus_agents.service.access")

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            # lifespan / websocket 不属于访问日志的职责
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        request_id = _sanitize_request_id(headers.get("x-request-id"))
        # Starlette 的 Request.state 就是 scope["state"] 的视图，写进去即对
        # 所有 handler 可见；setdefault 避免覆盖上游已放进去的共享状态。
        state = scope.setdefault("state", {})
        if isinstance(state, dict):
            state["request_id"] = request_id

        status_holder: list[int] = []

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder.append(message["status"])
                # 响应头里回写请求 ID：用户报障时「报一个 ID」就够
                headers_mut = list(message.get("headers", []))
                headers_mut.append((b"x-request-id", request_id.encode("latin-1")))
                message = {**message, "headers": headers_mut}
            await send(message)

        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            path = scope.get("path", "")
            if not path.startswith("/healthz"):
                status = status_holder[0] if status_holder else None
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                # 消息本身带上关键事实：JSON 管道按字段索引，人类看 tail -
                # f 日志时这行也能读。两种消费者都不用猜。
                self.logger.info(
                    "%s %s -> %s in %.1fms",
                    scope.get("method"),
                    path,
                    status,
                    duration_ms,
                    extra={
                        "request_id": request_id,
                        "method": scope.get("method"),
                        "path": path,
                        "status": status,
                        "duration_ms": duration_ms,
                    },
                )
