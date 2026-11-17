"""Observability tests — the log line is an API, treat it like one.

日志输出一旦被采集管道消费，字段名就是对外契约：改一个键名等于让所有历史
日志失去结构。所以这里钉死的是**形状**而不是「打印了点什么」：能不能被
``json.loads``、extras 有没有透传、异常有没有被折进字段、一行还是一行。
"""

from __future__ import annotations

import io
import json
import logging
import sys
from typing import Any

import httpx
import pytest

from conftest import add
from nexus_agents import ReActAgent
from nexus_agents.errors import ProviderTransportError
from nexus_agents.providers import ScriptedProvider
from nexus_agents.providers.base import GenerateRequest, LLMResponse
from nexus_agents.service import ServiceSettings, create_app
from nexus_agents.service.observability import (
    JsonFormatter,
    RequestLoggingMiddleware,
    configure_logging,
    current_request_id,
)

ACCESS_LOGGER = "nexus_agents.service.access"


# ================================================================ Formatter（纯函数）


def _record(
    message: str,
    level: int = logging.WARNING,
    exc_info: Any = None,
) -> logging.LogRecord:
    return logging.LogRecord(
        "nexus.test", level, __file__, 1, message, None, exc_info
    )


def test_json_formatter_renders_one_parseable_line_with_extras() -> None:
    record = _record("provider failed")
    record.request_id = "ab12cd34"
    record.status = 502

    line = JsonFormatter().format(record)

    # 契约一：一行一个对象。json.loads 能吃，且没有换行混进去
    payload = json.loads(line)
    assert "\n" not in line
    # 契约二：四个固定字段
    assert payload["event"] == "provider failed"
    assert payload["level"] == "warning"
    assert payload["logger"] == "nexus.test"
    assert "ts" in payload
    # 契约三：extra={...} 平铺进顶层，不需要 Formatter 认识字段名
    assert payload["request_id"] == "ab12cd34"
    assert payload["status"] == 502


def test_json_formatter_folds_exceptions_into_a_field() -> None:
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        record = _record("agent failed", level=logging.ERROR, exc_info=sys.exc_info())

    line = JsonFormatter().format(record)

    payload = json.loads(line)
    assert "\n" not in line  # 多行 traceback 拼进消息会把一行日志拆成好几行
    assert "RuntimeError: boom" in payload["exception"]


def test_json_formatter_folds_stack_info_into_a_field() -> None:
    record = _record("where am i", level=logging.DEBUG)
    record.stack_info = "  File ...\n    boom"

    payload = json.loads(JsonFormatter().format(record))

    assert "File" in payload["stack"]


# ================================================================ configure_logging


@pytest.mark.parametrize("json_output", [True, False])
def test_configure_logging_is_idempotent(json_output: bool) -> None:
    """重复配置不得堆 handler —— 否则生产里热重载一次，每行日志打两遍。"""
    stream = io.StringIO()
    name = "nexus._observability_test"
    try:
        configure_logging(
            "info", json_output=json_output, logger_name=name, stream=stream
        )
        configure_logging(
            "info", json_output=json_output, logger_name=name, stream=stream
        )
        log = logging.getLogger(name)
        log.info("hello", extra={"k": "v"})

        lines = [line for line in stream.getvalue().splitlines() if line]
        assert len(lines) == 1
        if json_output:
            payload = json.loads(lines[0])
            assert payload["event"] == "hello"
            assert payload["k"] == "v"
        else:
            assert "hello" in lines[0]
        # 自带 handler 后必须停止向 root 传播，否则 pytest 会看到重复记录
        assert log.propagate is False
    finally:
        log = logging.getLogger(name)
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()


def test_configure_logging_only_removes_its_own_handlers() -> None:
    """用户/框架挂上去的 handler 不归我们管，重复配置不许误删。"""
    name = "nexus._observability_test_foreign"
    log = logging.getLogger(name)
    foreign = logging.NullHandler()
    try:
        log.addHandler(foreign)
        configure_logging("info", json_output=False, logger_name=name, stream=io.StringIO())
        configure_logging("info", json_output=False, logger_name=name, stream=io.StringIO())
        assert foreign in log.handlers
        assert len(log.handlers) == 2  # 我们的 1 个 + 用户的 1 个
    finally:
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()


# ================================================================ 请求 ID 与访问日志


class TransportExplodingProvider:
    """拿真实异常类型测业务日志的关联字段。"""

    async def agenerate(self, request: GenerateRequest) -> LLMResponse:
        raise ProviderTransportError("upstream refused")


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_response_carries_a_generated_request_id() -> None:
    app = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]))

    async with _client(app) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    # token_hex(8)：16 个十六进制字符。不是断言格式漂亮，是断言它来自我们
    # 的生成器而不是某个中间层的默认值
    assert len(request_id) == 16
    int(request_id, 16)  # 全是 hex 字符，否则抛 ValueError


@pytest.mark.asyncio
async def test_client_supplied_request_id_is_echoed_and_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    app = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]))

    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER):
        async with _client(app) as client:
            response = await client.post(
                "/sessions",
                headers={"X-Request-ID": "my-trace-123"},
            )

    assert response.status_code == 201
    assert response.headers["x-request-id"] == "my-trace-123"
    records = [r for r in caplog.records if r.name == ACCESS_LOGGER]
    assert len(records) == 1
    record = records[0]
    assert record.request_id == "my-trace-123"
    assert record.path == "/sessions"
    assert record.method == "POST"
    assert record.status == 201
    assert record.duration_ms is not None


@pytest.mark.asyncio
async def test_healthz_is_exempt_from_access_logging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """探针每几秒一打，全记进日志 = 日志量与在线时长成正比、与流量无关。"""
    app = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]))

    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER):
        async with _client(app) as client:
            for _ in range(3):
                await client.get("/healthz")
            await client.post("/sessions")

    logged_paths = [r.path for r in caplog.records if r.name == ACCESS_LOGGER]
    assert logged_paths == ["/sessions"]


@pytest.mark.parametrize(
    "bad_id",
    ["with\nnewline", "with\rnewline", "x" * 200],
)
@pytest.mark.asyncio
async def test_hostile_request_ids_are_replaced(bad_id: str) -> None:
    """换行 = 日志注入；超长 = 头部/日志放大。两者都必须换成自己的 ID。"""
    app = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]))

    async with _client(app) as client:
        response = await client.get("/healthz", headers={"X-Request-ID": bad_id})

    echoed = response.headers["x-request-id"]
    assert echoed != bad_id
    assert "\n" not in echoed and "\r" not in echoed
    assert len(echoed) == 16


def test_current_request_id_reads_the_scope_state() -> None:
    assert current_request_id({"state": {"request_id": "r1"}}) == "r1"
    assert current_request_id({"state": {}}) is None
    assert current_request_id({}) is None
    # 非 dict 的 state（未来协议变化）不该把调用方炸掉
    assert current_request_id({"state": object()}) is None


@pytest.mark.asyncio
async def test_middleware_passes_non_http_scopes_through_untouched() -> None:
    """lifespan/websocket 不属于访问日志的职责，也不能被它拦住。"""
    seen: list[str] = []

    async def inner(scope: Any, receive: Any, send: Any) -> None:
        seen.append(scope["type"])

    middleware = RequestLoggingMiddleware(inner)  # type: ignore[arg-type]
    await middleware({"type": "lifespan"}, None, None)  # type: ignore[arg-type]

    assert seen == ["lifespan"]


@pytest.mark.asyncio
async def test_business_logs_correlate_with_the_request_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """provider 挂了的那行 warning 必须能靠 request_id 和访问日志对上账。"""
    app = create_app(ReActAgent(TransportExplodingProvider(), [add]))

    with caplog.at_level(logging.WARNING, logger="nexus_agents.service"):
        async with _client(app) as client:
            created = await client.post("/sessions", json={})
            session_id = created.json()["session_id"]
            failed = await client.post(
                f"/sessions/{session_id}/messages", json={"text": "hi"}
            )

    assert failed.status_code == 502
    warnings = [
        r
        for r in caplog.records
        if r.name == "nexus_agents.service" and r.levelno == logging.WARNING
    ]
    assert len(warnings) == 1
    assert warnings[0].request_id == failed.headers["x-request-id"]
    assert warnings[0].session_id == session_id


# ================================================================ 配置


def test_service_settings_rejects_bad_rate_limit_values() -> None:
    with pytest.raises(ValueError, match="rate_limit_per_minute"):
        ServiceSettings(rate_limit_per_minute=0)
    with pytest.raises(ValueError, match="rate_limit_burst"):
        ServiceSettings(rate_limit_burst=0)
