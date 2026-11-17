"""Rate limiting: a token bucket per caller, in ~120 honest lines.

为什么是令牌桶而不是「固定窗口计数」
------------------------------------
固定窗口在窗口边界会放行 **两倍** 流量：限 60/min，59 秒打满 60 个、
61 秒再打 60 个 —— 两秒内 120 个请求全部通过。令牌桶用「容量 + 匀速回填」
把突发预算和长期速率分开：容量就是允许的瞬时突发，回填率就是承诺的均值，
边界处不会翻倍。

为什么 key 用摘要而不是原文
---------------------------
调用方的身份是 ``X-API-Key``（没有密钥时退化为客户端地址）。密钥是凭据：
一旦原文进了内存表、又被 ``repr``/日志带出去，就是一次泄露。SHA-256 的前
16 个十六进制位做 key，桶行为完全一致，泄露面消失。

为什么桶要封顶
--------------
桶按调用方增长。公网服务每分钟来一万个陌生 IP，每个都建一个永不回收的桶，
限流器自己就成了内存放大器。``max_keys`` 到顶时按 LRU 淘汰最久未活跃的桶
—— 被淘汰的调用方下一请求会拿到新桶（满的），这是可接受的保守行为。
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable

from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["TokenBucket", "RateLimiter", "RateLimitMiddleware", "client_identity"]


# ================================================================ 令牌桶


class TokenBucket:
    """Capacity ``burst``, refilled at ``rate`` tokens per second. No locks.

    纯同步 + 可注入时钟：确定性测试不需要 ``sleep``，也不需要事件循环。
    单进程事件循环里 ``try_take`` 之间没有 await，天然原子；多 worker 部署
    必须换成 Redis 之类的共享实现 —— 那是 v0.7 的事，这里不假装支持。

    惰性回填：桶里存「上次取token的时刻」，取的时候一次性补齐中间应得的
    token，平时零开销。时钟可注入（``monotonic`` 默认），倒退的时钟按
    「不回填」处理，不会抛异常。
    """

    __slots__ = ("_clock", "capacity", "rate", "tokens", "updated")

    def __init__(
        self,
        capacity: float,
        rate: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if rate <= 0:
            raise ValueError("rate must be > 0")
        self.capacity = capacity
        self.rate = rate
        self.tokens = capacity
        self.updated = clock()
        self._clock = clock

    def try_take(self, count: float = 1.0) -> bool:
        """Take up to ``count`` tokens; return False without side effect if short."""
        now = self._clock()
        # 情况 1：时钟倒退（注入错误/系统调整）→ 间隔为负，max 兜底为 0
        elapsed = max(0.0, now - self.updated)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.updated = now
        if self.tokens >= count:
            self.tokens -= count
            return True
        return False

    def retry_after(self, count: float = 1.0) -> float:
        """Seconds until ``count`` tokens are available (0 if already enough)."""
        now = self._clock()
        elapsed = max(0.0, now - self.updated)
        available = min(self.capacity, self.tokens + elapsed * self.rate)
        if available >= count:
            return 0.0
        return math.ceil((count - available) / self.rate)


# ================================================================ 限流器


def client_identity(headers: dict[str, str], client_host: str | None) -> str:
    """Stable, secret-safe identity for one caller.

    有密钥按密钥的摘要（同一密钥在多个连接/地址后面仍是同一个调用方）；
    没密钥按客户端地址。地址可以伪造（X-Forwarded-For 之类我们不信任），
    所以这一层只做「公平与防失误」，不做安全边界 —— 安全边界是 api_key。
    """
    api_key = headers.get("x-api-key")
    if api_key:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:16]
        return f"key:{digest}"
    return f"host:{client_host or 'unknown'}"


class RateLimiter:
    """Buckets keyed by caller identity, bounded by ``max_keys`` (LRU)."""

    def __init__(
        self,
        *,
        requests_per_minute: float,
        burst: int = 10,
        max_keys: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        if max_keys < 1:
            raise ValueError("max_keys must be >= 1")
        self.rate = requests_per_minute / 60.0
        self.burst = burst
        self.max_keys = max_keys
        self._clock = clock
        # OrderedDict 只为 move_to_end/popitem 这对 LRU 原语，无并发考虑（见上）
        self._buckets: OrderedDict[str, TokenBucket] = OrderedDict()

    def check(self, identity: str) -> bool:
        """Consume one token for ``identity``; True means allowed."""
        bucket = self._buckets.get(identity)
        if bucket is None:
            if len(self._buckets) >= self.max_keys:
                # 淘汰最久未活跃的调用方，而不是最老的桶创建者：
                # 每次 check 都 move_to_end，所以队头就是最久没人来的
                self._buckets.popitem(last=False)
            bucket = TokenBucket(self.burst, self.rate, self._clock)
            self._buckets[identity] = bucket
        else:
            self._buckets.move_to_end(identity)
        return bucket.try_take()

    def retry_after(self, identity: str) -> float:
        """0 if a next request would pass now, else seconds to wait."""
        bucket = self._buckets.get(identity)
        if bucket is None:
            return 0.0
        return bucket.retry_after()


# ================================================================ ASGI 中间件


class RateLimitMiddleware:
    """429 with ``Retry-After`` before the request reaches the router.

    纯 ASGI、只拦 ``/sessions``：``/healthz`` 必须豁免 —— 编排器的探针频率
    由基础设施决定，不该吃业务配额；探针被限流会直接把健康的实例摘出
    负载均衡，一次「限流器打死探针」的事故就够了。
    """

    def __init__(
        self,
        app: ASGIApp,
        limiter: RateLimiter,
        *,
        prefix: str = "/sessions",
    ) -> None:
        self.app = app
        self.limiter = limiter
        self.prefix = prefix

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not scope.get("path", "").startswith(self.prefix):
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1")
            for k, v in scope.get("headers", [])
        }
        client = scope.get("client")
        identity = client_identity(headers, client[0] if client else None)
        if self.limiter.check(identity):
            await self.app(scope, receive, send)
            return
        wait = max(1, int(self.limiter.retry_after(identity)))
        body = b'{"detail":"rate limit exceeded; retry later"}'
        start: Message = {
            "type": "http.response.start",
            "status": 429,
            "headers": [
                (b"content-type", b"application/json"),
                (b"retry-after", str(wait).encode("latin-1")),
            ],
        }
        await send(start)
        await send({"type": "http.response.body", "body": body})
