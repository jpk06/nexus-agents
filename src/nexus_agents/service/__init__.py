"""Optional HTTP layer — ``pip install "nexus-agents[api]"``.

刻意不从顶层 :mod:`nexus_agents` 导入：FastAPI 是可选依赖，核心运行时不该
因为「有人只想用 Agent 循环」就被迫装一整套 web 框架。分包导入是这里
最省事的边界：``from nexus_agents import ReActAgent`` 不需要 FastAPI，
``from nexus_agents.service import create_app`` 才需要。
"""

from __future__ import annotations

from .app import ServiceSettings, create_app
from .sessions import InMemorySessionStore, Session, SessionStore
from .sse import SSE_HEADERS, SSE_MEDIA_TYPE, encode_comment, encode_event

__all__ = [
    "create_app",
    "ServiceSettings",
    "Session",
    "SessionStore",
    "InMemorySessionStore",
    "SSE_HEADERS",
    "SSE_MEDIA_TYPE",
    "encode_event",
    "encode_comment",
]
