"""The one tool that ships with the service.

为什么内置一个取时间的工具
--------------------------
新部署的人第一次提问时，如果服务里一个工具都没有，``/healthz`` 返回
``"tools": []``，SSE 流里永远只有 ``llm.call`` / ``llm.response`` ——
「工具调用链路到底是通的吗」这个问题只能靠读代码回答。

一个取当前时间的工具只要十几行，却能让第一次 ``curl`` 就看到完整的
``tool.call`` → ``tool.result`` → 最终答案，把「装配是否正确」变成可观察的
事实。它同时没有任何副作用和外部依赖，所以放进默认工具集是安全的。

为什么时区是固定偏移表，而不是 ``zoneinfo``
-------------------------------------------
``zoneinfo`` 在 slim 镜像里要额外装 tzdata；而这个工具的目的只是证明链路
通了，不是做日历计算。固定偏移表让镜像少一层系统依赖，代价是 DST（夏令时）
期间的伦敦/纽约时间会差一小时 —— 这个取舍写在 docstring 里，而不是藏着。
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from ..tools import tool

__all__ = ["BUILTIN_TOOLS", "current_time"]

#: 别名 -> UTC 偏移小时数。键用小写，查表前统一 lower()。
_OFFSETS: dict[str, int] = {
    "utc": 0,
    "gmt": 0,
    "london": 0,
    "beijing": 8,
    "shanghai": 8,
    "asia/shanghai": 8,
    "tokyo": 9,
    "asia/tokyo": 9,
    "newyork": -5,
    "america/new_york": -5,
}


@tool
def current_time(zone: str) -> str:
    """Return the current date and time for a timezone alias.

    Args:
        zone: timezone alias, one of utc, beijing, tokyo, london, newyork
    """
    offset = _OFFSETS.get(zone.strip().lower())
    if offset is None:
        # 返回「可选值清单」而不是抛异常：模型读到这句就知道下一步该怎么改，
        # 而抛异常要走一遍 ToolResult(is_error=True) 的回喂，多一轮 token。
        return f"unknown zone {zone!r}; supported zones: {', '.join(sorted(_OFFSETS))}"
    moment = datetime.now(UTC) + timedelta(hours=offset)
    return f"{moment.strftime('%Y-%m-%d %H:%M:%S')} (UTC{offset:+d})"


BUILTIN_TOOLS = (current_time,)
