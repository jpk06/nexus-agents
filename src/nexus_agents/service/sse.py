"""Server-Sent Events framing — five lines of spec, so no dependency needed.

为什么不引 sse-starlette
------------------------
SSE 的线格式只有一类东西：``field: value`` 加一个空行分隔。自己实现的好处
是它能被**纯函数单测**覆盖 —— 多行 ``data``、非法事件名、注释心跳这些边界
都能用一行断言钉死；引一个库只会多一层需要理解的抽象，还多一个要跟版本
的依赖。

一个容易被忽略的规则：``data`` 的值里如果出现换行，必须拆成多条 ``data:``
行，客户端会把它们用 ``\\n`` 重新拼起来。直接塞一个含换行的字符串，格式
就从这里开始烂掉，而且只在对端解析时才暴露。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

__all__ = ["SSE_MEDIA_TYPE", "SSE_HEADERS", "encode_event", "encode_comment"]

#: SSE 的媒体类型。写成常量而不是散落的字面量，避免某处手滑写成 text/json。
SSE_MEDIA_TYPE = "text/event-stream"

#: 反向代理的两个坑：nginx 默认会给上游响应做缓冲（事件全被攒起来，
#: 「流式」就变成了「最后一次性吐出来」），所以显式关掉缓冲。
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _check_single_line(value: str, *, field: str) -> str:
    """事件名/ID 里绝不能有换行 —— 否则等于伪造了额外的协议行。"""
    if not value:
        raise ValueError(f"SSE {field} must not be empty")
    if "\n" in value or "\r" in value:
        raise ValueError(f"SSE {field} must be a single line: {value!r}")
    return value


def encode_event(
    event: str,
    data: Mapping[str, Any] | str,
    *,
    event_id: str | None = None,
) -> str:
    """Encode one SSE frame.

    ``data`` 可以是字符串（原样发送，按行拆分）或映射（序列化成 JSON）。
    序列化用 ``ensure_ascii=False``：中文答案占 SSE 流的绝大多数字节，
    转义成 ``\\uXXXX`` 会让体积涨三倍，还没法用 curl 直接读。
    """
    lines: list[str] = []
    if event_id is not None:
        lines.append(f"id: {_check_single_line(event_id, field='event id')}")
    lines.append(f"event: {_check_single_line(event, field='event name')}")

    payload = data if isinstance(data, str) else json.dumps(
        data, ensure_ascii=False, separators=(",", ":")
    )
    # splitlines() 对空串返回 []，但一个空 data 行仍然要发出去
    for line in payload.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def encode_comment(text: str = "keep-alive") -> str:
    """A comment frame. 客户端会忽略它，中间设备却会因此重置空闲计时。

    这就是心跳的全部作用：不是为了传数据，是为了让 nginx / 负载均衡 /
    浏览器别把一条「安静了 30 秒」的连接当成死连接掐掉。LLM 的第一步
    常常要等好几秒，没有心跳就等着被掐。
    """
    body = "".join(f": {line}\n" for line in text.splitlines() or [""])
    return f"{body}\n"
