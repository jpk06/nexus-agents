"""Tests for the ReAct control loop."""

from __future__ import annotations

import pytest

from conftest import add, boom
from nexus_agents import AgentConfig, AgentError, MaxStepsExceeded, ReActAgent, ToolRegistry
from nexus_agents.memory import ConversationMemory
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.tracing import EventType
from nexus_agents.types import Message, Role, ToolCall, assistant_message, user_message


def echo_after_tool_call(tool_name: str, arguments: dict, answer: str):
    """A scripted policy: call one tool, then answer using its result."""

    def policy(request) -> Message | str:
        saw_tool_result = any(m.role is Role.TOOL for m in request.messages)
        if saw_tool_result:
            return answer
        return tool_call_message(tool_name, arguments)

    return policy


def agent_with(provider, tools=(add,), **config) -> ReActAgent:
    return ReActAgent(provider, list(tools), config=AgentConfig(**config))


# ---------------------------------------------------------------- 基本流程


def test_answers_without_tools():
    provider = ScriptedProvider(["42"])
    result = agent_with(provider).run("What is the answer?")

    assert result.answer == "42"
    assert result.succeeded is True
    assert result.stopped_reason.value == "final"
    assert result.steps == 1
    assert provider.call_count == 1


def test_calls_tool_then_answers():
    provider = ScriptedProvider([echo_after_tool_call("add", {"a": 2, "b": 3}, "5")])
    result = agent_with(provider).run("2 + 3?")

    assert result.answer == "5"
    assert result.steps == 2
    assert len(result.tool_calls) == 1
    # 工具结果必须真的进了对话，模型才看得见
    tool_messages = [m for m in result.messages if m.role is Role.TOOL]
    assert tool_messages[0].content == "5"
    assert tool_messages[0].name == "add"


def test_message_order_is_strictly_chronological():
    provider = ScriptedProvider([echo_after_tool_call("add", {"a": 1, "b": 1}, "2")])
    result = agent_with(provider).run("sum")

    roles = [m.role for m in result.messages]
    assert roles == [Role.USER, Role.ASSISTANT, Role.TOOL, Role.ASSISTANT]
    # 工具消息必须紧跟在发起调用的 assistant 消息之后
    assert result.messages[2].tool_call_id == result.messages[1].tool_calls[0].id


def test_system_prompt_is_sent_first():
    provider = ScriptedProvider(["ok"])
    agent_with(provider).run("hi", system="You are terse.")

    assert provider.last_request.messages[0].role is Role.SYSTEM
    assert provider.last_request.messages[0].content == "You are terse."


def test_config_system_prompt_is_used_when_no_override():
    provider = ScriptedProvider(["ok"])
    agent_with(provider, system_prompt="be brief").run("hi")

    assert provider.last_request.messages[0].content == "be brief"


def test_tools_are_forwarded_to_the_provider():
    provider = ScriptedProvider(["ok"])
    agent_with(provider).run("hi")

    assert [t.name for t in provider.last_request.tools] == ["add"]


# ---------------------------------------------------------------- 失败与恢复


def test_unknown_tool_error_is_fed_back_and_agent_recovers():
    provider = ScriptedProvider(
        [
            tool_call_message("multiply", {"a": 2, "b": 3}),  # 工具不存在
            echo_after_tool_call("add", {"a": 2, "b": 3}, "5"),
        ]
    )
    result = agent_with(provider).run("2*3?")

    assert result.answer == "5"
    errors = [e for e in result.trace if e.payload.get("is_error")]
    assert len(errors) == 1
    assert "multiply" in errors[0].payload["tool"]


def test_raising_tool_does_not_crash_the_run():
    provider = ScriptedProvider(
        [tool_call_message("boom"), echo_after_tool_call("add", {"a": 1, "b": 1}, "ok")]
    )
    result = agent_with(provider, tools=(add, boom)).run("go")

    assert result.succeeded is True
    assert any("kaboom" in m.content for m in result.messages if m.role is Role.TOOL)


def test_bad_arguments_are_reported_back_to_the_model():
    provider = ScriptedProvider([tool_call_message("add", {"a": "x", "b": 1}), "gave up"])
    result = agent_with(provider).run("add")

    tool_message = next(m for m in result.messages if m.role is Role.TOOL)
    assert tool_message.is_error is True
    assert "a" in tool_message.content


def test_stop_on_tool_error_aborts_immediately():
    provider = ScriptedProvider([tool_call_message("boom")])
    result = agent_with(provider, tools=(boom,), stop_on_tool_error=True).run("go")

    assert result.succeeded is False
    assert result.stopped_reason.value == "tool_error"
    assert result.steps == 1
    assert "kaboom" in result.answer


def test_consecutive_error_limit_stops_the_run():
    calls = [tool_call_message("boom", {}, call_id=f"c{i}") for i in range(5)]
    provider = ScriptedProvider(calls)
    result = agent_with(provider, tools=(boom,), max_consecutive_errors=2).run("go")

    assert result.stopped_reason.value == "tool_error"
    assert len([e for e in result.trace if e.type == EventType.TOOL_RESULT]) == 2


# ---------------------------------------------------------------- 步数上限


def test_max_steps_is_enforced():
    provider = ScriptedProvider([tool_call_message("add", {"a": 1, "b": 1})])
    result = agent_with(provider, max_steps=3).run("loop forever")

    assert result.stopped_reason.value == "max_steps"
    assert result.steps == 3
    assert result.succeeded is False
    assert provider.call_count == 3


def test_max_steps_falls_back_to_last_assistant_text():
    # 模型一边说话一边继续调工具，直到被叫停
    stuck = assistant_message(
        "partial", tool_calls=(ToolCall(id="c1", name="add", arguments={"a": 1, "b": 1}),)
    )
    provider = ScriptedProvider([stuck])
    result = agent_with(provider, max_steps=4).run("loop")

    assert result.stopped_reason.value == "max_steps"
    assert result.answer == "partial"


def test_config_rejects_non_positive_max_steps():
    with pytest.raises(ValueError):
        AgentConfig(max_steps=0)


# ---------------------------------------------------------------- 观测与成本


def test_usage_accumulates_across_steps():
    from nexus_agents.types import Usage

    provider = ScriptedProvider(
        [echo_after_tool_call("add", {"a": 1, "b": 2}, "3")],
        usage=Usage(prompt_tokens=10, completion_tokens=5),
    )
    result = agent_with(provider).run("sum")

    assert result.usage.prompt_tokens == 20
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 30


def test_trace_covers_the_whole_run():
    provider = ScriptedProvider([echo_after_tool_call("add", {"a": 1, "b": 2}, "3")])
    result = agent_with(provider).run("sum")

    types = [e.type for e in result.trace]
    assert types[0] == EventType.AGENT_START
    assert types[-1] == EventType.AGENT_END
    assert EventType.TOOL_CALL in types
    assert EventType.TOOL_RESULT in types
    assert EventType.LLM_CALL in types


def test_llm_call_events_record_duration():
    provider = ScriptedProvider(["done"])
    result = agent_with(provider).run("hi")

    llm_events = [e for e in result.trace if e.type == EventType.LLM_CALL]
    assert len(llm_events) == 1
    assert llm_events[0].duration_ms is not None


# ---------------------------------------------------------------- 结果 API


def test_raise_for_status_passes_on_success():
    provider = ScriptedProvider(["done"])
    result = agent_with(provider).run("hi")
    assert result.raise_for_status() is result


def test_raise_for_status_raises_max_steps():
    provider = ScriptedProvider([tool_call_message("add", {"a": 1, "b": 1})])
    result = agent_with(provider, max_steps=2).run("loop")
    with pytest.raises(MaxStepsExceeded):
        result.raise_for_status()


def test_raise_for_status_raises_agent_error_on_tool_failure():
    provider = ScriptedProvider([tool_call_message("boom")])
    result = agent_with(provider, tools=(boom,), stop_on_tool_error=True).run("go")
    with pytest.raises(AgentError):
        result.raise_for_status()


def test_result_serialises_to_dict():
    provider = ScriptedProvider([echo_after_tool_call("add", {"a": 1, "b": 2}, "3")])
    payload = agent_with(provider).run("sum").to_dict()

    assert payload["answer"] == "3"
    assert payload["stopped_reason"] == "final"
    assert payload["usage"]["total_tokens"] >= 0
    assert isinstance(payload["trace"], list)


def test_agent_accepts_a_registry_directly():
    provider = ScriptedProvider([echo_after_tool_call("add", {"a": 1, "b": 2}, "3")])
    agent = ReActAgent(provider, ToolRegistry([add]))
    assert agent.tools == (add,)
    assert agent.run("sum").answer == "3"


def test_run_is_repeatable_and_stateless():
    provider = ScriptedProvider([echo_after_tool_call("add", {"a": 1, "b": 2}, "3")])
    agent = agent_with(provider)
    first = agent.run("sum")
    provider.reset()
    second = agent.run("sum")

    assert first.answer == second.answer
    assert first.steps == second.steps


def test_repr_shows_tools():
    agent = ReActAgent(ScriptedProvider([]), [add], name="calc")
    assert "calc" in repr(agent)
    assert "add" in repr(agent)


# ---------------------------------------------------------------- history 注入


def test_history_is_sent_between_system_prompt_and_new_question():
    """逐字历史的位置必须固定：system → 历史 → 新问题。

    位置错了不会报错，模型只会「看起来有点傻」—— 这种 bug 只能靠断言顺序
    抓出来。
    """
    provider = ScriptedProvider(["第二个答案"])
    agent = agent_with(provider)
    history = [user_message("第一个问题"), assistant_message("第一个答案")]

    agent.run(
        "第二个问题",
        system="你是库管",
        history=history,
    )

    sent = provider.last_request.messages
    assert [(m.role, m.content) for m in sent] == [
        (Role.SYSTEM, "你是库管"),
        (Role.USER, "第一个问题"),
        (Role.ASSISTANT, "第一个答案"),
        (Role.USER, "第二个问题"),
    ]


def test_history_and_memory_can_be_combined():
    """历史负责「接得上」，记忆负责「想得起」—— 两者叠加而不是互相覆盖。"""
    provider = ScriptedProvider(["答案"])
    agent = agent_with(provider)
    memory = ConversationMemory()
    memory.remember_turn("去年的同类问题", "去年的答案")

    agent.run(
        "今年的同类问题",
        memory=memory,
        history=[user_message("刚才说了什么"), assistant_message("刚才的答案")],
    )

    sent = provider.last_request.messages
    kinds = [m.role for m in sent]
    # 先召回记忆（system），再逐字历史，最后是新问题
    assert kinds == [Role.SYSTEM, Role.USER, Role.ASSISTANT, Role.USER]
    assert "去年的答案" in sent[0].content
    assert sent[1].content == "刚才说了什么"
    assert sent[-1].content == "今年的同类问题"


def test_history_defaults_to_nothing():
    provider = ScriptedProvider(["答案"])
    agent_with(provider).run("只有一个问题")
    assert [m.role for m in provider.last_request.messages] == [Role.USER]
