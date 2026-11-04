"""Tests for tool definition, validation, execution and the registry."""

from __future__ import annotations

import pytest

from conftest import add, boom, convert, describe, find, repeat
from nexus_agents import ToolRegistry, ToolSpec, tool
from nexus_agents.errors import DuplicateToolError, ToolNotFoundError
from nexus_agents.types import ToolCall


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"id_{name}", name=name, arguments=arguments)


# ---------------------------------------------------------------- 定义


def test_tool_keeps_name_and_docstring_summary():
    assert add.name == "add"
    assert add.description == "Add two integers."
    assert add.parameters["required"] == ["a", "b"]


def test_decorated_function_is_still_callable():
    # 装饰不能破坏原有调用方式，否则老的单测会全挂
    assert add(a=1, b=2) == 3


def test_tool_decorator_with_explicit_metadata():
    @tool(name="sum_up", description="Adds stuff.")
    def fn(x: int) -> int:
        return x

    assert fn.name == "sum_up"
    assert fn.description == "Adds stuff."


def test_build_tool_spec_is_idempotent():
    spec = ToolSpec(
        name="add", description="d", parameters={}, fn=lambda: None
    )
    assert tool(spec) is spec


def test_api_payload_shape():
    payload = add.to_api_payload()
    assert payload["type"] == "function"
    assert payload["function"]["name"] == "add"
    assert payload["function"]["parameters"]["type"] == "object"


# ---------------------------------------------------------------- 校验与强转


def test_arguments_are_coerced_from_model_strings():
    result = add.invoke(call("add", a="7", b="8"))
    assert result.is_error is False
    assert result.output == "15"


def test_enum_argument_is_coerced():
    result = convert.invoke(call("convert", value=5, unit="inch"))
    assert result.output == "5.0 -> inch"


def test_defaults_are_applied_when_omitted():
    result = convert.invoke(call("convert", value=5))
    assert result.output == "5.0 -> cm"


def test_literal_argument_is_validated():
    assert find.invoke(call("find", items=[1, 2], mode="min")).output == "1"
    bad = find.invoke(call("find", items=[1, 2], mode="nope"))
    assert bad.is_error is True
    assert "mode" in bad.error


def test_unexpected_argument_is_rejected():
    result = add.invoke(call("add", a=1, b=2, c=3))
    assert result.is_error is True
    assert "extra" in result.error.lower()


def test_missing_required_argument_is_rejected():
    result = add.invoke(call("add", a=1))
    assert result.is_error is True
    assert "b" in result.error


def test_optional_argument_defaults_to_none():
    assert describe.invoke(call("describe")).output == "nothing"
    assert describe.invoke(call("describe", target="cup")).output == "cup"


def test_keyword_only_argument_is_supported():
    assert repeat.invoke(call("repeat", text="hi", times=3, sep="-")).output == "hi-hi-hi"


def test_non_string_results_are_serialised():
    @tool
    def stats() -> dict:
        """Return a mapping."""
        return {"n": 1}

    assert stats.invoke(call("stats")).output == '{"n": 1}'


# ---------------------------------------------------------------- 错误隔离


def test_raising_tool_becomes_error_result_not_exception():
    result = boom.invoke(call("boom"))
    assert result.is_error is True
    assert "kaboom" in result.error
    assert result.name == "boom"
    assert result.tool_call_id == "id_boom"


def test_exception_type_is_preserved_in_message():
    assert "ValueError" in boom.invoke(call("boom")).error


# ---------------------------------------------------------------- 注册中心


def test_registry_registers_and_lists_tools(calc_tools):
    assert set(calc_tools.names) == {"add", "convert", "repeat", "boom", "find", "describe"}
    assert len(calc_tools) == 6
    assert "add" in calc_tools
    assert calc_tools.get("add") is add


def test_duplicate_registration_is_rejected(calc_tools):
    with pytest.raises(DuplicateToolError):
        calc_tools.register(add)


def test_unknown_tool_lookup_raises(calc_tools):
    with pytest.raises(ToolNotFoundError) as exc:
        calc_tools.get("nope")
    assert "add" in str(exc.value)  # 报错里带上可用工具，方便模型自救


def test_registry_invokes_unknown_tool_as_error(calc_tools):
    result = calc_tools.invoke(call("nope"))
    assert result.is_error is True
    assert result.name == "nope"


def test_registry_dispatches_to_registered_tool(calc_tools):
    assert calc_tools.invoke(call("add", a=1, b=2)).output == "3"


def test_with_tools_returns_new_registry(calc_tools):
    @tool
    def extra() -> str:
        """Extra tool."""
        return "x"

    combined = calc_tools.with_tools(extra)
    assert len(combined) == 7
    assert len(calc_tools) == 6  # 原注册中心不受影响


def test_restricted_to_limits_the_tool_set(calc_tools):
    limited = calc_tools.restricted_to(["add"])
    assert list(limited.names) == ["add"]
    assert len(calc_tools) == 6


def test_restricted_to_rejects_unknown_name(calc_tools):
    with pytest.raises(ToolNotFoundError):
        calc_tools.restricted_to(["missing"])


def test_registry_accepts_plain_functions():
    def plain(x: int) -> int:
        """Double it."""
        return x * 2

    registry = ToolRegistry([plain])
    assert registry.invoke(call("plain", x=4)).output == "8"


def test_registry_payload(calc_tools):
    payload = calc_tools.to_api_payload()
    assert len(payload) == 6
    assert all(p["type"] == "function" for p in payload)
