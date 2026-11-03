"""Supervisor + workers: three agents coordinated by one state graph.

Offline and deterministic — run it with::

    python examples/multiagent.py

多智能体最容易写成一团互相调用的函数。这里换成图：

- **supervisor** 只看状态、决定下一步派谁干活（真正的实现里这一步可以
  交给一个 LLM，这里用确定性规则代替，方便看清流程）
- **worker** 是一个普通的 `ReActAgent`，完全不知道自己被编排
- **回边**：审校不通过就退回写作，改够了就收手（无限返工是最贵的失败模式）

```
supervisor ──► research ──┐
     ▲                    │
     └────────────────────┘
     ├──► write ──► review ──► (rejected) write
     └──► END
```
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus_agents import (
    END,
    AgentConfig,
    InMemoryCheckpointer,
    ReActAgent,
    StateGraph,
    ToolRegistry,
    append,
    sum_values,
    tool,
)
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.types import Role

MAX_REVISIONS = 3


# ---------------------------------------------------------------- 工具与 Agent


@tool
def search_archive(query: str) -> str:
    """Search the Jingdezhen archive.

    Args:
        query: what to look up
    """
    return "celadon glaze: iron oxide 1.5%, fired at 1280C in reduction atmosphere"


def make_researcher() -> ReActAgent:
    def policy(request):
        if any(m.role is Role.TOOL for m in request.messages):
            return "celadon glaze = iron oxide 1.5%, 1280C, reduction firing."
        return tool_call_message("search_archive", {"query": "celadon glaze"})

    return ReActAgent(
        ScriptedProvider([policy], usage=None),
        ToolRegistry([search_archive]),
        config=AgentConfig(max_steps=4),
        name="researcher",
    )


def make_writer() -> ReActAgent:
    def policy(request):
        return "Celadon owes its colour to iron oxide (1.5%) fired at 1280C."

    return ReActAgent(
        ScriptedProvider([policy], usage=None),
        config=AgentConfig(max_steps=2),
        name="writer",
    )


def make_reviewer() -> ReActAgent:
    """Approves from the second revision onwards — enough to show the loop."""

    def policy(request):
        prompt = request.messages[-1].content
        if "revision 2" in prompt or f"revision {MAX_REVISIONS}" in prompt:
            return "approved: sources cited, figures present."
        return "rejected: needs the firing temperature and the iron ratio."

    return ReActAgent(
        ScriptedProvider([policy], usage=None),
        config=AgentConfig(max_steps=2),
        name="reviewer",
    )


# ---------------------------------------------------------------- 节点


def supervisor_node(state):
    """Decide who works next. Returns no update — routing is its only job."""
    return {}


def route(state) -> str:
    if not state.get("research"):
        return "research"
    if not state.get("draft"):
        return "write"
    # 草稿被改过之后必须重新审，否则会拿着旧结论一路返工
    if state.get("reviewed_round") != state.get("round"):
        return "review"
    if state["verdict"].startswith("approved"):
        return END
    if state.get("round", 0) >= MAX_REVISIONS:
        return END  # 改够了就收手，不无限返工
    return "write"


def build_graph(researcher, writer, reviewer):
    graph = StateGraph({"log": append, "tokens": sum_values})

    graph.add_node("supervisor", supervisor_node)

    def research(state):
        result = researcher.run(state["task"])
        return {"research": result.answer, "tokens": result.usage.total_tokens, "log": "research"}

    def write(state):
        result = writer.run(f"Write a short paragraph using: {state['research']}")
        return {
            "draft": result.answer,
            "round": state.get("round", 0) + 1,
            "tokens": result.usage.total_tokens,
            "log": f"write#{state.get('round', 0) + 1}",
        }

    def review(state):
        result = reviewer.run(
            f"Review this draft (revision {state.get('round', 1)}): {state['draft']}"
        )
        return {
            "verdict": result.answer,
            "reviewed_round": state.get("round", 1),
            "tokens": result.usage.total_tokens,
            "log": f"review#{state.get('round', 1)}",
        }

    graph.add_node("research", research)
    graph.add_node("write", write)
    graph.add_node("review", review)

    graph.add_conditional_edges(
        "supervisor",
        route,
        {"research": "research", "write": "write", "review": "review"},
    )
    # 每个 worker 干完都回到 supervisor，由它决定下一步
    for worker in ("research", "write", "review"):
        graph.add_edge(worker, "supervisor")

    graph.set_entry_point("supervisor")
    return graph


def main() -> None:
    graph = build_graph(make_researcher(), make_writer(), make_reviewer())
    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store, max_steps=15, name="writing-team")

    result = compiled.invoke(
        {"task": "Explain why Jingdezhen celadon looks green."}, thread_id="team-1"
    )

    print("--- mermaid ---")
    print(graph.to_mermaid())

    print("\n--- run ---")
    print("status :", result.status.value)
    print("steps  :", result.steps)
    print("path   :", " -> ".join(result.path))
    print("revisions:", result.state.get("round"))
    print("verdict:", result.state.get("verdict"))
    print("draft  :", result.state.get("draft"))
    print("tokens :", result.state.get("tokens"))

    print("\n--- log ---")
    for line in result.state["log"]:
        print("  ", line)

    print("\n--- 每个节点耗时 ---")
    for node, ms in result.node_durations().items():
        print(f"  {node:10} {ms:.3f} ms")


if __name__ == "__main__":
    main()
