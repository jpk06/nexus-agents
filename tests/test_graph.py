"""Tests for the state-graph orchestration layer."""

from __future__ import annotations

import itertools
import json

import pytest

from nexus_agents import (
    END,
    AgentConfig,
    Checkpoint,
    CheckpointError,
    DuplicateNodeError,
    GraphError,
    GraphStatus,
    GraphStepLimitExceeded,
    InMemoryCheckpointer,
    InvalidRouteError,
    MissingEntryPointError,
    NodeNotFoundError,
    ReActAgent,
    StateGraph,
    ThreadNotFoundError,
    UnreachableNodeError,
    append,
    apply_update,
    keep_first,
    max_value,
    merge,
    replace,
    sum_values,
)
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.tracing import EventType

# ---------------------------------------------------------------- 辅助


def inc(key: str = "n", by: int = 1):
    """A node that bumps a counter."""

    def node(state):
        return {key: state.get(key, 0) + by}

    node.__name__ = f"inc_{key}"
    return node


def linear_graph(*names: str) -> StateGraph:
    graph = StateGraph()
    for name in names:
        graph.add_node(name, inc(name))
    for source, target in itertools.pairwise(names):
        graph.add_edge(source, target)
    graph.set_entry_point(names[0])
    return graph


# ---------------------------------------------------------------- 线性执行


def test_runs_nodes_in_order():
    result = linear_graph("a", "b", "c").compile().invoke({"n": 0})

    assert result.path == ("a", "b", "c")
    assert result.steps == 3
    assert result.status is GraphStatus.COMPLETED
    assert result.succeeded is True
    assert dict(result.state) == {"n": 0, "a": 1, "b": 1, "c": 1}


def test_single_node_graph_finishes_implicitly():
    graph = StateGraph()
    graph.add_node("only", inc("only")).set_entry_point("only")

    result = graph.compile().invoke()

    assert result.path == ("only",)
    assert result.state["only"] == 1


def test_finish_point_is_an_edge_to_end():
    graph = StateGraph()
    graph.add_node("a", inc("a"))
    graph.add_node("b", inc("b"))
    graph.add_edge("a", "b").set_finish_point("b").set_entry_point("a")

    compiled = graph.compile()
    result = compiled.invoke()

    assert result.path == ("a", "b")
    # 显式终点之后不再有节点，步数应该正好是 2
    assert result.steps == 2


def test_entry_point_can_be_set_last():
    graph = StateGraph()
    graph.add_node("a", inc("a"))
    graph.set_entry_point("a")

    assert graph.entry_point == "a"
    assert graph.nodes == ("a",)


def test_nodes_receive_a_read_only_view():
    def mutating(state):
        state["n"] = 99
        return {}

    graph = StateGraph()
    graph.add_node("bad", mutating).set_entry_point("bad")

    result = graph.compile().invoke({"n": 1})

    assert result.status is GraphStatus.ERROR
    assert result.path == ()  # 失败的节点不算走完
    assert "does not support item assignment" in (result.error or "")
    with pytest.raises(GraphError):
        result.raise_for_status()


def test_node_returning_none_leaves_state_untouched():
    graph = StateGraph()
    graph.add_node("noop", lambda state: None).set_entry_point("noop")

    result = graph.compile().invoke({"a": 1})

    assert dict(result.state) == {"a": 1}


def test_initial_state_is_never_mutated():
    initial = {"n": 0}
    linear_graph("a").compile().invoke(initial)

    assert initial == {"n": 0}


# ---------------------------------------------------------------- 条件路由


def test_conditional_edge_picks_a_branch():
    graph = StateGraph()
    graph.add_node("start", lambda state: {"score": 3})
    graph.add_node("high", lambda state: {"verdict": "high"})
    graph.add_node("low", lambda state: {"verdict": "low"})
    graph.add_conditional_edges(
        "start",
        lambda state: "high" if state["score"] >= 3 else "low",
        {"high": "high", "low": "low"},
    )
    graph.set_entry_point("start")

    result = graph.compile().invoke()

    assert result.path == ("start", "high")
    assert result.state["verdict"] == "high"


def test_conditional_edge_can_route_to_end():
    graph = StateGraph()
    graph.add_node("start", lambda state: {"tries": state.get("tries", 0) + 1})
    graph.add_node("again", lambda state: {"tries": state["tries"] + 1})
    graph.add_conditional_edges(
        "start", lambda state: "done" if state["tries"] >= 2 else "again",
        {"done": END, "again": "again"},
    )
    graph.add_edge("again", "start")
    graph.set_entry_point("start")

    result = graph.compile().invoke({"tries": 0})

    assert result.path == ("start", "again", "start")
    assert result.state["tries"] == 3


def test_router_may_return_a_node_name_without_mapping():
    graph = StateGraph()
    graph.add_node("a", lambda state: {"next": "b"})
    graph.add_node("b", lambda state: {})
    graph.add_conditional_edges("a", lambda state: state["next"])
    graph.set_entry_point("a")

    result = graph.compile().invoke()

    assert result.path == ("a", "b")


def test_unknown_branch_raises_invalid_route():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.add_conditional_edges("a", lambda state: "nowhere", {"ok": END})
    graph.set_entry_point("a")

    compiled = graph.compile()
    with pytest.raises(InvalidRouteError) as exc:
        compiled.invoke()

    assert "nowhere" in str(exc.value)


def test_a_node_cannot_have_both_static_and_conditional_edges():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.add_node("b", lambda state: {})
    graph.add_edge("a", "b")

    with pytest.raises(GraphError, match="static edge"):
        graph.add_conditional_edges("a", lambda state: END)


def test_end_cannot_have_outgoing_edges():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})

    with pytest.raises(GraphError):
        graph.add_edge(END, "a")


# ---------------------------------------------------------------- 循环与上限


def test_loop_runs_until_a_condition_holds():
    graph = StateGraph({"history": append})
    graph.add_node("search", lambda state: {"hits": state.get("hits", 0) + 1, "history": "search"})
    graph.add_conditional_edges(
        "search",
        lambda state: "search" if state["hits"] < 3 else "summarise",
        {"search": "search", "summarise": "summarise"},
    )
    graph.add_node("summarise", lambda state: {"summary": f"{state['hits']} hits"})
    graph.set_entry_point("search")

    result = graph.compile(max_steps=10).invoke()

    assert result.path == ("search", "search", "search", "summarise")
    assert result.state["summary"] == "3 hits"
    assert list(result.state["history"]) == ["search"] * 3


def test_step_limit_stops_the_run_and_reports_it():
    graph = StateGraph()
    graph.add_node("spin", lambda state: {"n": state.get("n", 0) + 1})
    graph.add_conditional_edges("spin", lambda state: "spin")
    graph.set_entry_point("spin")

    result = graph.compile(max_steps=4).invoke()

    assert result.status is GraphStatus.MAX_STEPS
    assert result.steps == 4
    assert result.succeeded is False
    with pytest.raises(GraphStepLimitExceeded) as exc:
        result.raise_for_status()
    assert "4" in str(exc.value)


def test_max_steps_of_one_runs_exactly_one_node():
    graph = StateGraph()
    graph.add_node("a", lambda state: {"a": 1})
    graph.add_node("b", lambda state: {"b": 1})
    graph.add_edge("a", "b").set_entry_point("a")

    result = graph.compile(max_steps=1).invoke()

    assert result.path == ("a",)
    assert result.status is GraphStatus.MAX_STEPS


def test_max_steps_must_be_positive():
    graph = StateGraph()
    graph.add_node("a", lambda state: {}).set_entry_point("a")

    with pytest.raises(ValueError):
        graph.compile(max_steps=0)


def test_node_exception_is_recorded_not_swallowed():
    def broken(state):
        raise RuntimeError("node exploded")

    graph = StateGraph()
    graph.add_node("a", lambda state: {"a": 1})
    graph.add_node("bad", broken)
    graph.add_edge("a", "bad")
    graph.add_node("never", lambda state: {"never": True})
    graph.add_conditional_edges("bad", lambda state: "never", {"never": "never"})
    graph.set_entry_point("a")

    result = graph.compile().invoke()

    assert result.status is GraphStatus.ERROR
    assert result.path == ("a",)
    assert result.exception is not None
    assert isinstance(result.exception, RuntimeError)
    assert result.events(EventType.ERROR)[0].payload["node"] == "bad"
    # 出错之后不再继续往下走
    assert "never" not in result.state


# ---------------------------------------------------------------- reducers


@pytest.mark.parametrize(
    ("reducer", "current", "incoming", "expected"),
    [
        (append, ["a"], "b", ["a", "b"]),
        (append, None, "b", ["b"]),
        (append, ["a"], ["b", "c"], ["a", "b", "c"]),
        (merge, {"a": 1}, {"b": 2}, {"a": 1, "b": 2}),
        (merge, None, {"b": 2}, {"b": 2}),
        (keep_first, "old", "new", "old"),
        (keep_first, None, "new", "new"),
        (sum_values, 2, 3, 5),
        (sum_values, None, 3, 3),
        (max_value, 2, 5, 5),
        (max_value, 5, 2, 5),
        (max_value, None, 2, 2),
    ],
)
def test_reducer_semantics(reducer, current, incoming, expected):
    assert reducer(current, incoming) == expected


def test_registered_reducer_accumulates_across_nodes():
    graph = StateGraph({"messages": append, "meta": merge, "total": sum_values})
    graph.add_node("a", lambda state: {"messages": "a", "meta": {"a": 1}, "total": 1})
    graph.add_node("b", lambda state: {"messages": "b", "meta": {"b": 2}, "total": 2})
    graph.add_edge("a", "b").set_entry_point("a")

    result = graph.compile().invoke()

    assert list(result.state["messages"]) == ["a", "b"]
    assert dict(result.state["meta"]) == {"a": 1, "b": 2}
    assert result.state["total"] == 3


def test_unregistered_field_is_last_write_wins():
    graph = StateGraph()
    graph.add_node("a", lambda state: {"value": "from a"})
    graph.add_node("b", lambda state: {"value": "from b"})
    graph.add_edge("a", "b").set_entry_point("a")

    assert graph.compile().invoke().state["value"] == "from b"


def test_apply_update_without_reducers_replaces():
    assert apply_update({"a": 1}, {"a": 2}) == {"a": 2}
    assert apply_update({"a": 1}, None) == {"a": 1}
    assert apply_update({"a": 1}, {"b": 2}) == {"a": 1, "b": 2}


# ---------------------------------------------------------------- 定义校验


def test_compiling_an_empty_graph_fails():
    with pytest.raises(GraphError, match="no nodes"):
        StateGraph().compile()


def test_compiling_without_entry_point_fails():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})

    with pytest.raises(MissingEntryPointError):
        graph.compile()


def test_unknown_entry_point_fails():
    graph = StateGraph()
    graph.add_node("a", lambda state: {}).set_entry_point("ghost")

    with pytest.raises(NodeNotFoundError):
        graph.compile()


def test_edge_to_unknown_node_fails():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.add_edge("a", "ghost").set_entry_point("a")

    with pytest.raises(NodeNotFoundError):
        graph.compile()


def test_duplicate_node_names_are_rejected():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})

    with pytest.raises(DuplicateNodeError):
        graph.add_node("a", lambda state: {})


def test_end_is_a_reserved_node_name():
    graph = StateGraph()
    with pytest.raises(GraphError, match="reserved"):
        graph.add_node(END, lambda state: {})


def test_node_must_be_callable():
    graph = StateGraph()
    with pytest.raises(TypeError):
        graph.add_node("a", "not callable")  # type: ignore[arg-type]


def test_unreachable_nodes_are_rejected():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.add_node("orphan", lambda state: {})
    graph.set_entry_point("a")

    with pytest.raises(UnreachableNodeError) as exc:
        graph.compile()

    assert "orphan" in str(exc.value)


def test_nodes_reachable_only_through_a_branch_are_ok():
    graph = StateGraph()
    graph.add_node("a", lambda state: {"go": "b"})
    graph.add_node("b", lambda state: {})
    graph.add_conditional_edges("a", lambda state: state["go"], {"b": "b", "done": END})
    graph.set_entry_point("a")

    assert graph.nodes == ("a", "b")  # compile 通过即证明可达性检查认了分支


# ---------------------------------------------------------------- checkpoint


def test_checkpoints_are_saved_after_every_node():
    store = InMemoryCheckpointer()
    compiled = linear_graph("a", "b", "c").compile(checkpointer=store)

    result = compiled.invoke({"n": 0}, thread_id="t1")

    assert [c.step for c in store.history("t1")] == [1, 2, 3]
    assert store.history("t1")[0].state["a"] == 1
    assert store.load("t1").finished is True  # 最后一条指向 None
    assert result.thread_id == "t1"


def test_resume_continues_without_rerunning_nodes():
    store = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("a", lambda state: {"log": "a"})
    graph.add_node("b", lambda state: {"log": "b"})
    graph.add_edge("a", "b").set_entry_point("a")
    compiled = graph.compile(checkpointer=store)

    first = compiled.invoke(thread_id="t1")
    assert first.path == ("a", "b")

    # 人为把线程回退到「a 已跑完、下一步是 b」的那一条
    store.clear("t1")
    store.save(Checkpoint(thread_id="t1", step=1, node="b", state={"log": "a"}))
    resumed = compiled.resume("t1")

    assert resumed.path == ("b",)
    assert resumed.state["log"] == "b"


def test_resume_requires_a_checkpointer():
    compiled = linear_graph("a").compile()

    with pytest.raises(CheckpointError):
        compiled.resume("nope")


def test_resume_unknown_thread_raises():
    compiled = linear_graph("a").compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(ThreadNotFoundError):
        compiled.resume("missing")


def test_get_state_without_checkpointer_returns_none():
    compiled = linear_graph("a").compile()

    assert compiled.get_state("t1") is None


def test_thread_id_is_generated_when_omitted():
    store = InMemoryCheckpointer()
    result = linear_graph("a").compile(checkpointer=store).invoke()

    assert result.thread_id.startswith("thread-")
    assert result.thread_id in store


def test_human_approval_can_be_resumed_from_a_checkpoint():
    """人工审批场景：流程停在 review 等人类点头，之后用同一 thread_id 续跑。"""
    store = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("draft", lambda state: {"draft": "v1"})
    graph.add_node("review", lambda state: {"reviewed": True})
    graph.add_node("publish", lambda state: {"published": True})
    graph.add_edge("draft", "review")
    graph.add_conditional_edges(
        "review",
        lambda state: "publish" if state.get("approved") else END,
        {"publish": "publish"},
    )
    graph.set_entry_point("draft")
    compiled = graph.compile(checkpointer=store)

    waiting = compiled.invoke(thread_id="doc-42")
    assert waiting.status is GraphStatus.COMPLETED
    assert waiting.path == ("draft", "review")
    assert "published" not in waiting.state
    assert store.load("doc-42").finished is True  # 停在「等待人类」这一步

    # 人类在外部把 approved 置为 True，写回同一条线程
    approved = dict(waiting.state) | {"approved": True}
    store.save(Checkpoint(thread_id="doc-42", step=2, node="review", state=approved))

    resumed = compiled.resume("doc-42")
    assert resumed.path == ("review", "publish")
    assert resumed.state["published"] is True
    assert resumed.steps == 4  # 前两步的账没有丢


# ---------------------------------------------------------------- 可观测性


def test_trace_records_the_whole_run():
    result = linear_graph("a", "b").compile().invoke()

    types = [e.type for e in result.trace]
    assert types[0] == EventType.GRAPH_START
    assert types[-1] == EventType.GRAPH_END
    assert types.count(EventType.NODE_START) == 2
    assert types.count(EventType.ROUTE) == 2
    assert result.events(EventType.NODE_END)[0].payload["updated"] == ["a"]


def test_checkpoint_events_are_emitted():
    compiled = linear_graph("a", "b").compile(checkpointer=InMemoryCheckpointer())

    result = compiled.invoke()

    assert len(result.events(EventType.CHECKPOINT)) == 2


def test_node_durations_are_measured():
    result = linear_graph("a", "b").compile().invoke()

    durations = result.node_durations()
    assert set(durations) == {"a", "b"}
    assert all(value >= 0 for value in durations.values())


def test_result_is_json_serialisable():
    result = linear_graph("a", "b").compile().invoke({"n": 1})

    payload = json.dumps(result.to_dict())
    assert json.loads(payload)["status"] == "completed"


def test_mermaid_rendering():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.add_node("b", lambda state: {})
    graph.add_conditional_edges("a", lambda state: "b", {"yes": "b", "no": END})
    graph.set_entry_point("a")

    text = graph.to_mermaid()

    assert "__start__ --> a" in text
    assert "a -->|yes| b" in text
    assert "a -->|no| __end__" in text


# ---------------------------------------------------------------- 与 Agent 组合


def test_a_node_can_run_an_agent():
    """节点里跑一个 ReActAgent —— 这才是这一层存在的意义。"""
    provider = ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"])

    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = ReActAgent(provider, [add], config=AgentConfig(max_steps=3))
    graph = StateGraph()
    graph.add_node("solve", lambda state: {"answer": agent.run(state["question"]).answer})
    graph.set_entry_point("solve")

    result = graph.compile().invoke({"question": "2+3?"})

    assert result.state["answer"] == "5"


def test_decorator_registers_a_node():
    graph = StateGraph()

    @graph.node("draft")
    def draft(state):
        return {"draft": "v1"}

    graph.set_entry_point("draft")

    result = graph.compile().invoke()

    assert result.state["draft"] == "v1"
    assert draft({"x": 1}) == {"draft": "v1"}  # 装饰后仍可直接调用


# ---------------------------------------------------------------- 边界与自省


def test_reducer_edge_cases():
    assert replace("old", "new") == "new"
    assert max_value(5, None) == 5


def test_raise_for_status_returns_self_on_success():
    result = linear_graph("a").compile().invoke()

    assert result.raise_for_status() is result


def test_empty_node_name_is_rejected():
    graph = StateGraph()
    with pytest.raises(GraphError, match="non-empty"):
        graph.add_node("", lambda state: {})


def test_conditional_node_cannot_also_get_a_static_edge():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.add_node("b", lambda state: {})
    graph.add_conditional_edges("a", lambda state: END)

    with pytest.raises(GraphError, match="conditional branch"):
        graph.add_edge("a", "b")


def test_end_cannot_get_a_conditional_edge():
    graph = StateGraph()
    with pytest.raises(GraphError):
        graph.add_conditional_edges(END, lambda state: "a")


def test_router_must_be_callable():
    graph = StateGraph()
    graph.add_node("a", lambda state: {})

    with pytest.raises(TypeError):
        graph.add_conditional_edges("a", "not callable")  # type: ignore[arg-type]


def test_builder_exposes_its_definition():
    graph = StateGraph({"log": append})
    graph.add_node("a", lambda state: {})
    graph.add_node("b", lambda state: {})
    graph.add_edge("a", "b").set_entry_point("a")

    assert "log" in graph.reducers
    assert "nodes=" in repr(graph)


def test_compiled_graph_exposes_its_definition():
    compiled = linear_graph("a", "b").compile(max_steps=7)

    assert compiled.nodes == ("a", "b")
    assert compiled.entry_point == "a"
    assert "max_steps=7" in repr(compiled)


def test_mermaid_renders_static_edges_and_dynamic_routers():
    static = StateGraph()
    static.add_node("a", lambda state: {})
    static.add_node("b", lambda state: {})
    static.add_edge("a", "b").set_entry_point("a")
    assert "  a --> b" in static.to_mermaid()

    dynamic = StateGraph()
    dynamic.add_node("a", lambda state: {})
    dynamic.add_conditional_edges("a", lambda state: "a").set_entry_point("a")
    assert "__dynamic__" in dynamic.to_mermaid()


def test_get_state_returns_the_stored_checkpoint():
    store = InMemoryCheckpointer()
    compiled = linear_graph("a", "b").compile(checkpointer=store)

    compiled.invoke(thread_id="t1")
    snapshot = compiled.get_state("t1")

    assert snapshot is not None
    assert snapshot.step == 2
    assert snapshot.finished is True
