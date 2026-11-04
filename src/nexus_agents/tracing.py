"""Lightweight, dependency-free tracing for agent runs.

为什么不用 OpenTelemetry
------------------------
本阶段的目标是「看得懂一次 run 到底发生了什么」，而不是接一个可观测性
平台。这里用一条扁平的事件流就够：可断言、可序列化、可打印，后面要接
OTel 也只是把事件转成 span 而已。
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

__all__ = ["EventType", "TraceEvent", "Tracer"]


class EventType:
    """Well-known event names emitted by the agent loop."""

    AGENT_START = "agent.start"
    AGENT_END = "agent.end"
    STEP_START = "step.start"
    LLM_CALL = "llm.call"
    LLM_RESPONSE = "llm.response"
    TOOL_CALL = "tool.call"
    TOOL_RESULT = "tool.result"
    ERROR = "error"
    MAX_STEPS = "agent.max_steps"
    # 状态图专用（graph.py）
    GRAPH_START = "graph.start"
    GRAPH_END = "graph.end"
    NODE_START = "node.start"
    NODE_END = "node.end"
    ROUTE = "graph.route"
    CHECKPOINT = "graph.checkpoint"


@dataclass(frozen=True)
class TraceEvent:
    """One immutable record of something that happened during a run."""

    type: str
    step: int = 0
    payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "step": self.step,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        }


class Tracer:
    """Collects :class:`TraceEvent` objects for a single agent run."""

    def __init__(self) -> None:
        self.events: list[TraceEvent] = []

    # ------------------------------------------------------------ 写入

    def emit(self, type: str, *, step: int = 0, **payload: Any) -> TraceEvent:
        event = TraceEvent(type=type, step=step, payload=payload)
        self.events.append(event)
        return event

    @contextmanager
    def span(self, type: str, *, step: int = 0, **payload: Any) -> Iterator[TraceEvent]:
        """Emit an event on enter and record its duration on exit."""
        start = time.perf_counter()
        event = TraceEvent(type=type, step=step, payload=payload)
        self.events.append(event)
        try:
            yield event
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            object.__setattr__(event, "duration_ms", round(elapsed, 3))

    # ------------------------------------------------------------ 读取

    def of_type(self, type: str) -> list[TraceEvent]:
        return [e for e in self.events if e.type == type]

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.to_dict() for e in self.events]

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self) -> Iterator[TraceEvent]:
        return iter(self.events)

    def __repr__(self) -> str:
        return f"Tracer({len(self.events)} events)"
