"""Tests for the built-in tracer."""

from __future__ import annotations

import pytest

from nexus_agents.tracing import EventType, TraceEvent, Tracer


def test_emit_records_event_with_payload():
    tracer = Tracer()
    tracer.emit(EventType.TOOL_CALL, step=2, tool="add")

    assert len(tracer) == 1
    event = tracer.events[0]
    assert event.type == EventType.TOOL_CALL
    assert event.step == 2
    assert event.payload == {"tool": "add"}
    assert isinstance(event, TraceEvent)


def test_span_records_duration():
    tracer = Tracer()
    with tracer.span(EventType.LLM_CALL, step=1, model="m") as event:
        pass

    assert event.duration_ms is not None
    assert event.duration_ms >= 0


def test_span_records_duration_even_when_body_raises():
    tracer = Tracer()
    with pytest.raises(RuntimeError), tracer.span(EventType.LLM_CALL):
        raise RuntimeError("boom")

    assert tracer.events[0].duration_ms is not None


def test_of_type_filters_events():
    tracer = Tracer()
    tracer.emit(EventType.TOOL_CALL)
    tracer.emit(EventType.TOOL_RESULT)
    tracer.emit(EventType.TOOL_CALL)

    assert len(tracer.of_type(EventType.TOOL_CALL)) == 2
    assert len(tracer.of_type(EventType.TOOL_RESULT)) == 1


def test_iteration_and_serialisation():
    tracer = Tracer()
    tracer.emit(EventType.AGENT_START, input="hi")

    assert [e.type for e in tracer] == [EventType.AGENT_START]
    payload = tracer.to_dict()
    assert payload[0]["payload"] == {"input": "hi"}
    assert payload[0]["timestamp"] > 0


def test_repr_shows_count():
    tracer = Tracer()
    tracer.emit("x")
    assert "1" in repr(tracer)


def test_event_to_dict_contains_all_fields():
    event = TraceEvent(type="t", step=1, payload={"a": 1})
    data = event.to_dict()
    assert set(data) == {"type", "step", "payload", "timestamp", "duration_ms"}
