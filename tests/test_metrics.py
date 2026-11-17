"""Metrics tests — 纯函数先钉死文本契约，再用 ASGI 层钉死采集行为.

值得逐条说明的三个断言选择：

1. **直方图桶断言「累积」而不是「各桶差值」**。Prometheus 的桶语义就是
   累积（``le=0.1`` 包含所有 ≤0.1 的观测）。写成差值等于按自己的直觉
   重新发明协议 —— 抓取器 sum 出来的数字会全部翻倍。
2. **路由标签断言模板而非原文**。UUID 进标签是势爆炸，这是指标层最
   容易「demo 没事、上线打爆」的一个坑，值得一条专门的测试。
3. **在流进行中断言 in_flight**。gauge 的意义就是「此刻」，等流结束再
   断言它等于 0，连「从来不加 1」的错误实现都测不出来。
"""

from __future__ import annotations

import asyncio

import pytest

from conftest import add
from nexus_agents.providers import ScriptedProvider
from nexus_agents.service import (
    DEFAULT_BUCKETS,
    PROMETHEUS_CONTENT_TYPE,
    Metrics,
    render,
)
from nexus_agents.service.metrics import MetricsMiddleware
from test_service import GatedProvider, build_app, client_for

_ = add  # conftest 里注册的工具，导入仅为保证 fixture 可用


# ================================================================ 注册表（纯函数）


def test_empty_registry_renders_all_three_families() -> None:
    text = render(Metrics())
    assert "# TYPE nexus_http_requests_total counter" in text
    assert "# TYPE nexus_http_request_duration_seconds histogram" in text
    assert "# TYPE nexus_http_requests_in_flight gauge" in text
    assert text.rstrip().endswith("nexus_http_requests_in_flight 0")


def test_counter_accumulates_per_route_and_status() -> None:
    metrics = Metrics()
    metrics.observe("/sessions", 201, 0.001)
    metrics.observe("/sessions", 201, 0.002)
    metrics.observe("/sessions", 409, 0.000)
    assert metrics.request_count("/sessions", 201) == 2
    assert metrics.request_count("/sessions", 409) == 1
    assert metrics.request_count("/sessions", 500) == 0
    text = render(metrics)
    assert 'nexus_http_requests_total{route="/sessions",status="201"} 2' in text
    assert 'nexus_http_requests_total{route="/sessions",status="409"} 1' in text


def test_histogram_buckets_are_cumulative_and_inf_equals_count() -> None:
    metrics = Metrics()
    # 0.004 落进 le=0.005；0.6 落进 le=1.0 及以下所有 <=0.6 的桶
    metrics.observe("/messages", 200, 0.004)
    metrics.observe("/messages", 200, 0.6)

    text = render(metrics)
    assert 'nexus_http_request_duration_seconds_bucket{route="/messages",le="0.005"} 1' in text
    assert 'nexus_http_request_duration_seconds_bucket{route="/messages",le="0.01"} 1' in text
    assert 'nexus_http_request_duration_seconds_bucket{route="/messages",le="0.5"} 1' in text
    assert 'nexus_http_request_duration_seconds_bucket{route="/messages",le="1.0"} 2' in text
    assert 'nexus_http_request_duration_seconds_bucket{route="/messages",le="+Inf"} 2' in text
    assert 'nexus_http_request_duration_seconds_count{route="/messages"} 2' in text
    # sum 保留全部精度，不做四舍五入 —— 平均数交给查询侧算
    assert 'nexus_http_request_duration_seconds_sum{route="/messages"} 0.604' in text


def test_label_values_are_escaped() -> None:
    metrics = Metrics()
    metrics.observe('we"ird\\path\ninjection', 200, 0.01)
    text = render(metrics)
    assert (
        'nexus_http_requests_total{route="we\\"ird\\\\path\\ninjection",status="200"} 1'
        in text
    )
    # 渲染结果必须是单行文本：转义失败会把一行日志变两行伪造指标
    for line in text.splitlines():
        assert line.startswith(("#", "nexus_"))


def test_render_is_deterministic() -> None:
    metrics = Metrics()
    metrics.observe("/a", 200, 0.01)
    metrics.observe("/b", 404, 0.02)
    assert render(metrics) == render(metrics)


@pytest.mark.parametrize(
    "buckets",
    [(), (0.1, 0.05), (-1.0,), (0.1, 0.0)],
)
def test_invalid_buckets_are_rejected(buckets: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        Metrics(buckets=buckets)


def test_default_buckets_are_ascending() -> None:
    assert list(DEFAULT_BUCKETS) == sorted(DEFAULT_BUCKETS)


# ================================================================ 中间件与端点


@pytest.mark.asyncio
async def test_metrics_endpoint_serves_prometheus_text() -> None:
    async with client_for(build_app(ScriptedProvider(["ok"]))) as client:
        response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"] == PROMETHEUS_CONTENT_TYPE
    assert "# TYPE nexus_http_requests_total counter" in response.text


@pytest.mark.asyncio
async def test_paths_are_recorded_as_route_templates() -> None:
    provider = ScriptedProvider(["7"])
    async with client_for(build_app(provider)) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        reply = await client.post(f"/sessions/{sid}/messages", json={"text": "3+4?"})
        assert reply.status_code == 200

        text = (await client.get("/metrics")).text
    # 标签是模板，不是含 UUID 的原始路径 —— 势必须有界
    expected = 'nexus_http_requests_total{route="/sessions/{session_id}/messages",status="200"} 1'
    assert expected in text
    assert sid not in text


@pytest.mark.asyncio
async def test_unmatched_paths_collapse_into_one_series() -> None:
    async with client_for(build_app(ScriptedProvider(["ok"]))) as client:
        assert (await client.get("/no/such/path")).status_code == 404
        assert (await client.get("/another/missing")).status_code == 404
        text = (await client.get("/metrics")).text
    assert 'nexus_http_requests_total{route="<unmatched>",status="404"} 2' in text


@pytest.mark.asyncio
async def test_healthz_and_metrics_are_not_recorded() -> None:
    async with client_for(build_app(ScriptedProvider(["ok"]))) as client:
        await client.get("/healthz")
        await client.get("/metrics")
        await client.get("/metrics")
        text = (await client.get("/metrics")).text
    assert "/healthz" not in text
    assert 'route="/metrics"' not in text


@pytest.mark.asyncio
async def test_rate_limited_requests_still_counted_as_429() -> None:
    # 指标层在限流层外：错误率必须包含被限流拒绝的请求，否则
    # requests_total 会天然低估「用户视角的失败率」
    app = build_app(ScriptedProvider(["ok"]), rate_limit_per_minute=1, rate_limit_burst=2)
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        first = await client.post(f"/sessions/{sid}/messages", json={"text": "hi"})
        second = await client.post(f"/sessions/{sid}/messages", json={"text": "again"})
        text = (await client.get("/metrics")).text
    assert first.status_code == 200
    assert second.status_code == 429
    # 429 发生在路由之前（限流层直接短路），scope 里还没有 route，
    # 所以落进 <unmatched>：标签仍有界，429 率照样可查 —— 只是别指望
    # 它告诉你是哪条路由被限的
    assert 'nexus_http_requests_total{route="<unmatched>",status="429"} 1' in text


@pytest.mark.asyncio
async def test_in_flight_gauge_tracks_a_stream_in_progress() -> None:
    provider = GatedProvider()
    app = build_app(provider)
    async with client_for(app) as client:
        sid = (await client.post("/sessions", json={})).json()["session_id"]
        streaming = asyncio.create_task(
            client.post(f"/sessions/{sid}/messages/stream", json={"text": "慢"})
        )
        await asyncio.wait_for(provider.started.wait(), timeout=2)

        metrics: Metrics = app.state.metrics  # type: ignore[attr-defined]
        assert metrics.in_flight == 1  # 流挂着时必须能看到它

        provider.release.set()
        response = await asyncio.wait_for(streaming, timeout=2)
        assert response.status_code == 200
        assert metrics.in_flight == 0  # 结束后必须归零
        # SSE 的时长要记到流结束，而不是 response.start 那一刻：
        # 在 start 处掐表会把长连接记成微秒级 —— 指标说谎比没有更糟
        assert metrics.request_count("/sessions/{session_id}/messages/stream", 200) == 1


def test_middleware_passes_through_non_http_scopes() -> None:
    # lifespan 等 scope 不是 http：指标层必须原样放行，而不是记一笔
    # status=0 的垃圾序列
    seen: list[str] = []

    async def app(scope, receive, send):  # type: ignore[no-untyped-def]
        seen.append(scope["type"])

    async def receive():  # type: ignore[no-untyped-def]
        raise AssertionError("should not be called")

    async def send(message):  # type: ignore[no-untyped-def]
        raise AssertionError("should not be called")

    metrics = Metrics()
    middleware = MetricsMiddleware(app, metrics=metrics)  # type: ignore[arg-type]
    asyncio.run(middleware({"type": "lifespan"}, receive, send))  # type: ignore[arg-type]
    assert seen == ["lifespan"]
    assert metrics.in_flight == 0
    assert render(metrics).count("status=") == 0
