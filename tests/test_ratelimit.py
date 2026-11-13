"""Rate limiting tests — deterministic clocks, no sleeps anywhere.

限流逻辑的 bug 几乎都在时间上：窗口边界、时钟回退、回填上限。用注入时钟
把每一种时间情况**摆好姿势**再断言，比 ``sleep(1)`` 等 100 ms 的测试快、
稳、而且测的才是真性质。
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
import pytest

from conftest import add
from nexus_agents import ReActAgent
from nexus_agents.providers import ScriptedProvider
from nexus_agents.service import (
    RateLimiter,
    RateLimitMiddleware,
    ServiceSettings,
    TokenBucket,
    create_app,
)
from nexus_agents.service.ratelimit import client_identity

ACCESS_LOGGER = "nexus_agents.service.access"


# ================================================================ 令牌桶


def test_token_bucket_allows_only_the_burst_before_refill() -> None:
    now = [1000.0]
    bucket = TokenBucket(capacity=3, rate=10.0, clock=lambda: now[0])  # 10 token/s

    assert [bucket.try_take() for _ in range(3)] == [True, True, True]
    assert bucket.try_take() is False  # 突发预算用完

    now[0] += 0.1  # 回填 1 个 token
    assert bucket.try_take() is True
    assert bucket.try_take() is False


def test_token_bucket_refill_is_capped_at_capacity() -> None:
    """挂机一小时不能攒出一万个 token —— 容量就是瞬时突发的硬上限。"""
    now = [0.0]
    bucket = TokenBucket(capacity=2, rate=100.0, clock=lambda: now[0])
    bucket.try_take()
    bucket.try_take()

    now[0] += 3600
    assert bucket.try_take() is True
    assert bucket.try_take() is True  # 也就回到容量，第 3 个没有
    assert bucket.try_take() is False


def test_token_bucket_survives_a_backwards_clock() -> None:
    """注入错误或 NTP 调整让时钟倒退：不抛异常、不白送 token。"""
    now = [100.0]
    bucket = TokenBucket(capacity=1, rate=5.0, clock=lambda: now[0])
    assert bucket.try_take() is True

    now[0] = 50.0  # 倒退
    assert bucket.try_take() is False


def test_token_bucket_retry_after_rounds_up() -> None:
    now = [0.0]
    bucket = TokenBucket(capacity=1, rate=1.0, clock=lambda: now[0])  # 1 token/s
    assert bucket.try_take() is True

    now[0] = 0.001  # 差 0.999s 攒回一个 token
    assert bucket.retry_after() == 1  # Retry-After 必须向上取整，宁可多等


def test_token_bucket_retry_after_is_zero_when_tokens_available() -> None:
    bucket = TokenBucket(capacity=5, rate=1.0, clock=lambda: 0.0)
    assert bucket.retry_after() == 0.0


@pytest.mark.parametrize(
    "capacity, rate",
    [(0, 1), (-1, 1), (1, 0), (1, -0.5)],
)
def test_token_bucket_rejects_nonsense_parameters(capacity: float, rate: float) -> None:
    with pytest.raises(ValueError):
        TokenBucket(capacity=capacity, rate=rate)


# ================================================================ 身份


def test_client_identity_hashes_the_api_key() -> None:
    identity = client_identity({"x-api-key": "hunter2"}, "10.0.0.1")

    # 原文绝不能出现在身份串里 —— 它会跟着桶表、日志一起扩散
    assert "hunter2" not in identity
    assert identity.startswith("key:")
    assert len(identity) == len("key:") + 16
    # 同一密钥稳定，不同密钥隔离
    assert identity == client_identity({"x-api-key": "hunter2"}, None)
    assert identity != client_identity({"x-api-key": "hunter3"}, None)


def test_client_identity_falls_back_to_the_host() -> None:
    assert client_identity({}, "10.0.0.9") == "host:10.0.0.9"
    assert client_identity({}, None) == "host:unknown"


# ================================================================ 限流器


def test_rate_limiter_isolates_callers() -> None:
    limiter = RateLimiter(requests_per_minute=60, burst=2, clock=lambda: 0.0)

    assert limiter.check("key:aaa") is True
    assert limiter.check("key:aaa") is True
    assert limiter.check("key:aaa") is False  # 自己的桶空了
    assert limiter.check("key:bbb") is True  # 别人的桶不受影响


def test_rate_limiter_retry_after_is_zero_for_unknown_callers() -> None:
    limiter = RateLimiter(requests_per_minute=60, burst=1, clock=lambda: 0.0)

    assert limiter.retry_after("never-seen") == 0.0


def test_rate_limiter_evicts_least_recently_active_key() -> None:
    """桶数量必须有上限，否则限流器自己就是内存放大器。"""
    limiter = RateLimiter(requests_per_minute=60, burst=1, max_keys=2, clock=lambda: 0.0)

    assert limiter.check("a") is True
    assert limiter.check("b") is True
    assert limiter.check("c") is True  # 挤掉 a（最久未活跃）

    # a 的桶被淘汰 → 新桶是满的 → 放行。宁可漏放不可永久封禁
    assert limiter.check("a") is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"requests_per_minute": 0},
        {"requests_per_minute": -1},
        {"requests_per_minute": 60, "burst": 0},
        {"requests_per_minute": 60, "max_keys": 0},
    ],
)
def test_rate_limiter_rejects_nonsense_parameters(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        RateLimiter(**kwargs)


# ================================================================ 中间件（注入时钟，全确定性）


def _app_with_limiter(
    limiter: RateLimiter,
) -> RateLimitMiddleware:
    inner = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]))
    return RateLimitMiddleware(inner, limiter)


def _client(app: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


@pytest.mark.asyncio
async def test_requests_beyond_the_burst_get_429_with_retry_after() -> None:
    clock = [0.0]
    limiter = RateLimiter(requests_per_minute=60, burst=2, clock=lambda: clock[0])
    app = _app_with_limiter(limiter)

    async with _client(app) as client:
        first = await client.post("/sessions")
        second = await client.post("/sessions")
        third = await client.post("/sessions")

    assert first.status_code == 201
    assert second.status_code == 201
    assert third.status_code == 429
    assert third.headers["retry-after"] == "1"
    assert third.json() == {"detail": "rate limit exceeded; retry later"}


@pytest.mark.asyncio
async def test_rate_limited_caller_recovers_after_refill() -> None:
    clock = [0.0]
    limiter = RateLimiter(requests_per_minute=60, burst=1, clock=lambda: clock[0])
    app = _app_with_limiter(limiter)

    async with _client(app) as client:
        assert (await client.post("/sessions")).status_code == 201
        assert (await client.post("/sessions")).status_code == 429

        clock[0] += 1.0  # 攒回一个 token
        assert (await client.post("/sessions")).status_code == 201


@pytest.mark.asyncio
async def test_healthz_is_exempt_from_rate_limiting() -> None:
    """探针吃业务配额 → 健康实例被限流 → 被编排器摘除。这是真实事故模式。"""
    limiter = RateLimiter(requests_per_minute=60, burst=1, clock=lambda: 0.0)
    app = _app_with_limiter(limiter)

    async with _client(app) as client:
        for _ in range(5):
            response = await client.get("/healthz")
            assert response.status_code == 200
        # 业务配额一个没花
        assert (await client.post("/sessions")).status_code == 201


@pytest.mark.asyncio
async def test_no_rate_limiting_by_default() -> None:
    app = create_app(ReActAgent(ScriptedProvider(["hi"]), [add]))

    async with _client(app) as client:
        for _ in range(5):
            assert (await client.post("/sessions")).status_code == 201


@pytest.mark.asyncio
async def test_429s_are_visible_in_the_access_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """限流层在日志层里面：被挡的请求必须出现在访问日志里（status=429）。"""
    app = create_app(
        ReActAgent(ScriptedProvider(["hi"]), [add]),
        settings=ServiceSettings(rate_limit_per_minute=60, rate_limit_burst=1),
    )

    with caplog.at_level(logging.INFO, logger=ACCESS_LOGGER):
        async with _client(app) as client:
            await client.post("/sessions")
            rejected = await client.post("/sessions")

    assert rejected.status_code == 429
    # 完整装配里日志层在限流层外面：429 也带请求 ID，也在访问日志里
    assert len(rejected.headers["x-request-id"]) == 16
    statuses = [r.status for r in caplog.records if r.name == ACCESS_LOGGER]
    assert 429 in statuses
