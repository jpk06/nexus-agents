"""Tests for the message primitives every layer depends on."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from nexus_agents.types import (
    FinishReason,
    Message,
    Role,
    ToolCall,
    ToolResult,
    Usage,
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)

# ---------------------------------------------------------------- ToolCall


def test_arguments_given_as_json_string_are_decoded():
    call = ToolCall(id="1", name="add", arguments='{"a": 1, "b": 2}')
    assert call.arguments == {"a": 1, "b": 2}


def test_malformed_json_arguments_degrade_to_empty():
    assert ToolCall(id="1", name="add", arguments="{not json").arguments == {}


def test_non_object_json_is_wrapped():
    assert ToolCall(id="1", name="f", arguments="[1, 2]").arguments == {"value": [1, 2]}


def test_none_arguments_become_empty_dict():
    assert ToolCall(id="1", name="f", arguments=None).arguments == {}


def test_tool_call_is_immutable():
    call = ToolCall(id="1", name="add", arguments={})
    with pytest.raises(ValidationError):
        call.name = "other"


# ---------------------------------------------------------------- Usage


def test_usage_totals_and_addition():
    usage = Usage(prompt_tokens=3, completion_tokens=4)
    assert usage.total_tokens == 7
    combined = usage + Usage(prompt_tokens=1, completion_tokens=1)
    assert (combined.prompt_tokens, combined.completion_tokens) == (4, 5)


# ---------------------------------------------------------------- ToolResult


def test_failure_helper_marks_error():
    result = ToolResult.failure(call_id="c1", name="boom", error="nope")
    assert result.is_error is True
    assert result.error == "nope"
    assert result.output == ""


def test_tool_result_is_immutable():
    result = ToolResult(tool_call_id="c", name="n")
    with pytest.raises(ValidationError):
        result.output = "changed"


# ---------------------------------------------------------------- Message


def test_constructors_set_roles():
    assert system_message("s").role is Role.SYSTEM
    assert user_message("u").role is Role.USER
    assert assistant_message("a").role is Role.ASSISTANT


def test_tool_result_message_carries_error_flag():
    failure = ToolResult.failure(call_id="c9", name="t", error="bad")
    message = tool_result_message(failure)

    assert message.role is Role.TOOL
    assert message.tool_call_id == "c9"
    assert message.name == "t"
    assert message.is_error is True
    assert message.content == "bad"


def test_has_tool_calls():
    plain = assistant_message("hi")
    with_call = assistant_message(tool_calls=(ToolCall(id="1", name="t"),))
    assert plain.has_tool_calls is False
    assert with_call.has_tool_calls is True


def test_api_payload_for_plain_message():
    assert user_message("hello").to_api_payload() == {
        "role": "user",
        "content": "hello",
    }


def test_api_payload_serialises_tool_calls_as_json_strings():
    message = assistant_message(
        tool_calls=(ToolCall(id="c1", name="add", arguments={"a": 1}),)
    )
    payload = message.to_api_payload()

    assert payload["tool_calls"][0]["type"] == "function"
    assert payload["tool_calls"][0]["function"]["name"] == "add"
    assert json.loads(payload["tool_calls"][0]["function"]["arguments"]) == {"a": 1}


def test_api_payload_includes_tool_call_id():
    failure = ToolResult.failure(call_id="c5", name="t", error="x")
    payload = tool_result_message(failure).to_api_payload()
    assert payload["tool_call_id"] == "c5"
    assert payload["name"] == "t"


def test_message_is_immutable():
    message = user_message("hi")
    with pytest.raises(ValidationError):
        message.content = "bye"


def test_finish_reason_values_match_api():
    assert FinishReason.TOOL_CALLS.value == "tool_calls"
    assert FinishReason.STOP.value == "stop"


def test_message_round_trips_through_json():
    message = assistant_message("hi", tool_calls=(ToolCall(id="1", name="t"),))
    restored = Message.model_validate_json(message.model_dump_json())
    assert restored == message
