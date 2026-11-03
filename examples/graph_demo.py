"""State graph demo: a research loop that retries until the evidence is good.

Offline and deterministic — run it with::

    python examples/graph_demo.py

演示三件事：

1. **循环 + 条件边**：检索结果不够就换个关键词再检索，够了才去写结论
2. **checkpoint**：每一步都落盘，随时可以从某个 thread 接着跑
3. **可视化**：图能导出成 Mermaid，直接贴进文档
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus_agents import END, InMemoryCheckpointer, StateGraph, append

#: 假数据：不同关键词能命中的资料数量不同
CORPUS = {
    "celadon": ["celadon glaze recipe"],
    "kiln": ["dragon kiln temperature", "kiln atmosphere"],
    "glaze": ["iron oxide ratio", "glaze cracking"],
}
#: 每个关键词只能查一次，逼出「换关键词重试」的循环
QUERIES = ["celadon", "kiln", "glaze"]


def search(state):
    """Search with the next unused keyword and record the hits."""
    used = state.get("used", [])
    query = next(q for q in QUERIES if q not in used)
    hits = CORPUS[query]
    # evidence 注册了 append reducer，所以这里只返回「新增」的部分
    return {
        "used": [*used, query],
        "evidence": hits,
        "log": f"search({query}) -> {len(hits)}",
    }


def enough(state):
    """Router: keep searching until we have at least three pieces of evidence."""
    if len(state.get("evidence", [])) >= 3:
        return "write"
    if len(state.get("used", [])) >= len(QUERIES):
        return END  # 资料查完了还不够 -> 如实放弃，不要硬编
    return "search"


def write(state):
    evidence = state.get("evidence", [])
    return {
        "report": f"Found {len(evidence)} sources: " + "; ".join(evidence),
        "log": "write",
    }


def build_graph():
    graph = StateGraph({"evidence": append, "log": append})
    graph.add_node("search", search)
    graph.add_node("write", write)
    graph.add_conditional_edges(
        "search", enough, {"search": "search", "write": "write"}
    )
    graph.set_entry_point("search")
    return graph


def main() -> None:
    graph = build_graph()
    store = InMemoryCheckpointer()
    compiled = graph.compile(checkpointer=store, max_steps=8, name="research-loop")

    result = compiled.invoke({"evidence": [], "used": []}, thread_id="demo-1")

    print("--- mermaid ---")
    print(graph.to_mermaid())

    print("\n--- run ---")
    print("status    :", result.status.value)
    print("steps     :", result.steps)
    print("path      :", " -> ".join(result.path))
    print("report    :", result.state.get("report"))
    print("log       :", list(result.state["log"]))

    print("\n--- checkpoints (thread=demo-1) ---")
    for snapshot in store.history("demo-1"):
        count = len(snapshot.state["evidence"])
        print(f"  step {snapshot.step}  next={snapshot.node}  evidence={count}")

    print("\n--- node timings ---")
    for node, ms in result.node_durations().items():
        print(f"  {node:8} {ms:.3f} ms")

    # 步数用完时已经跑过的部分不会丢 —— 换个更大的预算就能接着跑完
    limited = graph.compile(checkpointer=store, max_steps=2, name="research-loop")
    partial = limited.invoke({"evidence": [], "used": []}, thread_id="demo-2")
    print(f"\n--- 只给 2 步 ---\nstatus : {partial.status.value}")
    print("path   :", " -> ".join(partial.path))

    continued = graph.compile(checkpointer=store, max_steps=8).resume("demo-2")
    print(f"\n--- 续跑 ---\nstatus : {continued.status.value}")
    print("path   :", " -> ".join(continued.path))
    print("report :", continued.state.get("report"))


if __name__ == "__main__":
    main()
