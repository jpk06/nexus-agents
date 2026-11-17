"""Metrics: a tiny Prometheus text-exposition registry, zero dependencies.

为什么自己写而不是用 ``prometheus_client``
------------------------------------------
核心仓库的哲学是**零运行时依赖**（FastAPI 是 ``[api]`` extra 才装的）。
``prometheus_client`` 很成熟，但把整个对象树（REGISTRY、进程收集器、线程
锁）拖进一个演示服务，换来的能力里我们只用得上「counter + histogram +
文本输出」。手写 ~180 行，换来三件事：无新依赖、行为完全可预测（没有
全局单例和后台线程）、测试不需要 ``REGISTRY.unregister`` 那套清理仪式。

为什么标签用路由模板而不是原始路径
----------------------------------
``/sessions/3f9a.../messages`` 里那个 UUID 是标签的**势**（cardinality）
杀手：每个新会话都让指标多一条时间序列，一个公网服务的 Prometheus 就被
我们打爆了。所以中间件在路由完成后读 ``scope["route"].path`` —— 拿到的
是 ``/sessions/{session_id}/messages`` 这种模板，序列数 = 路由数 × 状态码
数，天然有界。没匹配到路由的请求（404，以及**在路由之前**就被中间件
拒绝的 429 —— 限流层短路时 scope 里还没有 route）统一折叠成
``<unmatched>``：原始路径同样无界，而这类请求本来也只关心「有多少、
什么状态码」，不关心「打的是哪」。

为什么和限流器一样「只诚实支持单进程」
--------------------------------------
计数器是普通 dict 上的 ``+=``。单进程事件循环里 ``observe`` 之间没有
await，天然原子；一旦上多 worker，每个副本各数各的，``requests_total``
就成了「某一个副本的局部真相」。Prometheus 抓取侧可以按 instance 聚合
（sum by），所以这**不是错**，但必须在文档里说清楚 —— 和 TokenBucket
一样，这里不假装支持。

文本格式契约见 ``render``：``# HELP`` / ``# TYPE`` / 样本行，抓取器按
``text/plain; version=0.0.4`` 解析。这是 Prometheus 的开放标准，不需要
客户端 SDK 也能用 curl 验证。
"""

from __future__ import annotations

import time
from collections.abc import Mapping

from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["Metrics", "MetricsMiddleware", "render", "PROMETHEUS_CONTENT_TYPE"]

#: /metrics 响应的 Content-Type。抓取器靠 version 参数选择解析器；
#: 省略它时部分版本会退回旧格式假设，导致 +Inf 桶被当普通样本。
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

#: 直方图桶边界（秒）。沿用 prometheus_client 的默认桶 —— 它们是对
#: Web 请求延迟的长期经验值；这里没有真实流量数据，自造一组只会更差。
#: 约束：升序、全正数、最后一个由 ``render`` 合成 +Inf。
DEFAULT_BUCKETS: tuple[float, ...] = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
)

#: 既不计入也不暴露自身流量的路径：探针每几秒一次，记进序列只会把
#: 真实信号稀释掉（和访问日志跳过 /healthz 是同一个理由）。
_SKIPPED_PATH_PREFIXES = ("/healthz", "/metrics")


def _escape_label(value: str) -> str:
    """Prometheus 标签值的转义：``\\``、``"``、换行。"""
    return (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
    )


class _Histogram:
    """Cumulative buckets + sum + count. 一个标签组合一个实例。"""

    __slots__ = ("bucket_counts", "buckets", "count", "sum")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        # _cumulative[i] = 落在 buckets[0..=i] 区间的观测数；+Inf 在 render 合成
        self.buckets = buckets
        self.bucket_counts = [0] * len(buckets)
        self.count = 0
        self.sum = 0.0

    def observe(self, value: float) -> None:
        self.count += 1
        self.sum += value
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                # 累积桶：命中上界的同时，所有更低的界也 +1
                self.bucket_counts[i] += 1


class Metrics:
    """请求级指标的注册表。无锁、无后台线程、无全局单例。

    只有三个指标，每个都回答一个排障时会真的问的问题：

    * ``nexus_http_requests_total{route,status}`` —— 谁在多大比例上
      失败（4xx/5xx 占比 = 错误预算烧掉的速度）；
    * ``nexus_http_request_duration_seconds``（直方图）—— 慢在哪个路由、
      慢到什么程度（P99 用桶插值，不求全量精确）；
    * ``nexus_http_requests_in_flight``（gauge）—— 此刻挂着多少请求
      （含未完成的 SSE 长连接），这是背压的第一个信号。
    """

    def __init__(self, *, buckets: tuple[float, ...] = DEFAULT_BUCKETS) -> None:
        if not buckets or any(b <= 0 for b in buckets):
            raise ValueError("buckets must be non-empty and positive")
        if list(buckets) != sorted(buckets):
            raise ValueError("buckets must be ascending")
        self._buckets = buckets
        self._requests: dict[tuple[str, int], int] = {}
        self._durations: dict[str, _Histogram] = {}
        self.in_flight = 0

    # -------------------------------------------------- 记录侧（中间件调用）

    def observe(self, route: str, status: int, seconds: float) -> None:
        """Record one finished request. 标签值转义推迟到 render —— 记录侧
        保持「存的就是业务值」，转义纯粹是输出格式的私事。"""
        self._requests[(route, status)] = self._requests.get((route, status), 0) + 1
        histogram = self._durations.get(route)
        if histogram is None:
            histogram = self._durations[route] = _Histogram(self._buckets)
        histogram.observe(seconds)

    def request_started(self) -> None:
        self.in_flight += 1

    def request_finished(self) -> None:
        self.in_flight -= 1

    # -------------------------------------------------- 快照（测试与路由用）

    def request_count(self, route: str, status: int) -> int:
        return self._requests.get((route, status), 0)


def render(metrics: Metrics) -> str:
    """Serialize to Prometheus text-exposition format.

    输出顺序固定（counter → histogram → gauge，标签内按插入序）：同一份
    数据两次抓取的文本逐字节相同 —— 测试可以精确断言，diff 工具也不会
    因为顺序抖动报假警。
    """
    lines: list[str] = []
    lines.append("# HELP nexus_http_requests_total Total HTTP requests handled.")
    lines.append("# TYPE nexus_http_requests_total counter")
    for (route, status), count in metrics._requests.items():
        lines.append(
            f'nexus_http_requests_total{{route="{_escape_label(route)}",'
            f'status="{status}"}} {count}'
        )
    lines.append(
        "# HELP nexus_http_request_duration_seconds HTTP request duration in seconds."
    )
    lines.append("# TYPE nexus_http_request_duration_seconds histogram")
    for route, histogram in metrics._durations.items():
        route_label = f'route="{_escape_label(route)}"'
        for bound, bucket_count in zip(
            histogram.buckets, histogram.bucket_counts, strict=True
        ):
            lines.append(
                f"nexus_http_request_duration_seconds_bucket{{{route_label},"
                f'le="{bound}"}} {bucket_count}'
            )
        lines.append(
            f"nexus_http_request_duration_seconds_bucket{{{route_label},"
            f'le="+Inf"}} {histogram.count}'
        )
        lines.append(
            f"nexus_http_request_duration_seconds_sum{{{route_label}}}"
            f" {histogram.sum}"
        )
        lines.append(
            f"nexus_http_request_duration_seconds_count{{{route_label}}}"
            f" {histogram.count}"
        )
    lines.append("# HELP nexus_http_requests_in_flight Requests currently being served.")
    lines.append("# TYPE nexus_http_requests_in_flight gauge")
    lines.append(f"nexus_http_requests_in_flight {metrics.in_flight}")
    return "\n".join(lines) + "\n"


def _route_label(scope: Mapping[str, object]) -> str:
    """Prefer the route template over the raw path (see module docstring).

    必须在路由**之后**读：中间件包在路由外面，进来时 scope 里还没有
    route。这个函数只在响应已开始（``finally``）时被调用，那时 Starlette
    已把匹配到的 ``Route`` 塞进了 scope。
    """
    route = scope.get("route")
    path = getattr(route, "path", None)
    if isinstance(path, str) and path:
        return path
    return "<unmatched>"


class MetricsMiddleware:
    """Pure-ASGI metrics collection: wrap ``send``, record in ``finally``.

    和访问日志中间件同构（包 ``send`` 抓状态码，不用 ``BaseHTTPMiddleware``
    —— 那层包装和流式响应打架，理由见 observability.py）。延迟量到
    ``finally`` 才记：SSE 的响应要等最后一个 body 块发完，在
    ``http.response.start`` 处掐表会把长连接记成微秒级 —— 指标说谎比
    没有指标更糟。
    """

    def __init__(self, app: ASGIApp, metrics: Metrics) -> None:
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        if path.startswith(_SKIPPED_PATH_PREFIXES):
            await self.app(scope, receive, send)
            return
        status_holder: list[int] = []

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder.append(message["status"])
            await send(message)

        self.metrics.request_started()
        started = time.perf_counter()
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self.metrics.observe(
                route=_route_label(scope),
                status=status_holder[0] if status_holder else 0,
                seconds=time.perf_counter() - started,
            )
            self.metrics.request_finished()
