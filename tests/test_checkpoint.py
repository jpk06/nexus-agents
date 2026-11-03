"""Tests for checkpoint storage and semantics."""

from __future__ import annotations

import pytest

from nexus_agents import Checkpoint, InMemoryCheckpointer, StateGraph
from nexus_agents.checkpoint import Checkpointer, frozen_view


def make_checkpoint(thread_id: str = "t", step: int = 1, node: str | None = "b", **state):
    return Checkpoint(thread_id=thread_id, step=step, node=node, state=state)


# ---------------------------------------------------------------- Checkpoint 本身


def test_checkpoint_state_is_frozen():
    snapshot = make_checkpoint(x=1)

    with pytest.raises(TypeError):
        snapshot.state["x"] = 2  # type: ignore[index]


def test_checkpoint_metadata_defaults_to_empty():
    assert dict(make_checkpoint().metadata) == {}


def test_checkpoint_knows_whether_the_run_finished():
    assert make_checkpoint(node=None).finished is True
    assert make_checkpoint(node="b").finished is False


def test_checkpoint_round_trips_through_a_dict():
    snapshot = make_checkpoint(thread_id="t1", step=3, node=None, a=1, b="two")

    data = snapshot.to_dict()

    assert data["thread_id"] == "t1"
    assert data["step"] == 3
    assert data["node"] is None
    assert data["state"] == {"a": 1, "b": "two"}
    assert data["created_at"] == snapshot.created_at


def test_checkpoint_repr_is_readable():
    assert "t1" in repr(make_checkpoint(thread_id="t1", step=2))


def test_frozen_view_copies_the_mapping():
    source = {"a": 1}
    view = frozen_view(source)
    source["b"] = 2

    assert "b" not in view
    assert dict(view) == {"a": 1}


# ---------------------------------------------------------------- InMemoryCheckpointer


def test_saves_and_loads_the_latest_checkpoint():
    store = InMemoryCheckpointer()
    store.save(make_checkpoint(step=1, node="b", n=1))
    store.save(make_checkpoint(step=2, node=None, n=2))

    latest = store.load("t")

    assert latest is not None
    assert latest.step == 2
    assert latest.state["n"] == 2


def test_load_unknown_thread_returns_none():
    assert InMemoryCheckpointer().load("nope") is None


def test_history_is_oldest_first():
    store = InMemoryCheckpointer()
    for step in range(1, 4):
        store.save(make_checkpoint(step=step))

    assert [c.step for c in store.history("t")] == [1, 2, 3]


def test_history_of_unknown_thread_is_empty():
    assert InMemoryCheckpointer().history("nope") == ()


def test_at_step_finds_a_specific_snapshot():
    store = InMemoryCheckpointer()
    store.save(make_checkpoint(step=1, n=1))
    store.save(make_checkpoint(step=2, n=2))

    assert store.at_step("t", 1).state["n"] == 1  # type: ignore[union-attr]
    assert store.at_step("t", 9) is None


def test_threads_lists_every_thread_once():
    store = InMemoryCheckpointer()
    store.save(make_checkpoint(thread_id="a"))
    store.save(make_checkpoint(thread_id="a"))
    store.save(make_checkpoint(thread_id="b"))

    assert store.threads() == ("a", "b")


def test_clear_accepts_a_thread_or_everything():
    store = InMemoryCheckpointer()
    store.save(make_checkpoint(thread_id="a"))
    store.save(make_checkpoint(thread_id="b"))

    store.clear("a")
    assert store.threads() == ("b",)

    store.clear()
    assert store.threads() == ()


def test_clear_unknown_thread_is_a_noop():
    store = InMemoryCheckpointer()
    store.clear("ghost")

    assert len(store) == 0


def test_len_counts_every_checkpoint():
    store = InMemoryCheckpointer()
    store.save(make_checkpoint(thread_id="a"))
    store.save(make_checkpoint(thread_id="b"))

    assert len(store) == 2
    assert "a" in store
    assert repr(store).startswith("InMemoryCheckpointer(")


def test_in_memory_checkpointer_satisfies_the_protocol():
    assert isinstance(InMemoryCheckpointer(), Checkpointer)


# ---------------------------------------------------------------- 与图集成


def test_graph_writes_one_checkpoint_per_node():
    store = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("a", lambda state: {"a": 1})
    graph.add_node("b", lambda state: {"b": 1})
    graph.add_edge("a", "b").set_entry_point("a")

    graph.compile(checkpointer=store).invoke(thread_id="run-1")

    assert len(store.history("run-1")) == 2
    assert store.at_step("run-1", 1).node == "b"  # type: ignore[union-attr]
    assert store.load("run-1").node is None  # type: ignore[union-attr]


def test_checkpoint_carries_graph_metadata():
    store = InMemoryCheckpointer()
    graph = StateGraph()
    graph.add_node("a", lambda state: {})
    graph.set_entry_point("a")

    graph.compile(checkpointer=store, name="demo").invoke(thread_id="run-1")

    assert store.load("run-1").metadata["graph"] == "demo"  # type: ignore[index]
