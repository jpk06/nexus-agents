"""State-graph orchestration: nodes, conditional edges, loops and checkpoints.

为什么要有这一层
----------------
单个 ReAct 循环能做「一件事」，但真实任务往往是这样的：

1. 先检索，判断资料够不够
2. 不够就换个关键词再检索（**循环**）
3. 够了就交给写作 Agent，写完再交给审校 Agent（**顺序 + 分支**）
4. 审校不通过就退回写作（**回边**）

把这些写进 ``if/elif`` 里也能跑，但很快会变成没人敢改的意大利面。
状态图把「流程」显式化：节点是纯函数，边是数据，于是流程可以
被检查（有没有孤立的节点）、被可视化、被单测、被中断后恢复。

核心约定
--------
1. **状态是普通 dict**，节点只读它，返回**局部更新**而不是整个新状态。
   这样节点之间互不认识，只通过约定的字段通信。
2. **默认最后写入生效**，需要累积的字段（如消息列表）显式注册 reducer。
   这避免了一个隐蔽 bug：两个节点都写 ``messages`` 时互相覆盖。
3. **节点拿到的是只读视图**。直接改状态会当场 ``TypeError``，
   而不是悄悄影响后面的节点。
4. **跑满步数或节点抛异常都如实记录**，不会假装有结果。
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .checkpoint import Checkpoint, Checkpointer, frozen_view
from .errors import (
    CheckpointError,
    DuplicateNodeError,
    GraphError,
    GraphStepLimitExceeded,
    InvalidRouteError,
    MissingEntryPointError,
    NodeNotFoundError,
    ThreadNotFoundError,
    UnreachableNodeError,
)
from .tracing import EventType, TraceEvent, Tracer

__all__ = [
    "END",
    "GraphStatus",
    "GraphResult",
    "StateGraph",
    "CompiledGraph",
    "apply_update",
    # reducers
    "replace",
    "append",
    "merge",
    "keep_first",
    "sum_values",
    "max_value",
]

#: 终点哨兵。指向它的边表示这条分支结束。
END = "__end__"

#: 图的状态：一个普通字典。刻意不用类 —— 序列化、调试、合并都更简单。
State = Mapping[str, Any]
#: 节点返回值：要合并进状态的**局部**更新（``None`` 表示什么都不改）。
StateUpdate = Mapping[str, Any] | None
#: 节点：读状态，返回局部更新。
NodeFn = Callable[[State], StateUpdate]
#: 路由函数：读状态，返回分支名（节点名或 ``END``）。
RouteFn = Callable[[State], str]
#: Reducer：把「旧值 + 新值」合成一个值。
Reducer = Callable[[Any, Any], Any]


# ---------------------------------------------------------------- reducers


def replace(_current: Any, incoming: Any) -> Any:
    """Last write wins (the default)."""
    return incoming


def append(current: Any, incoming: Any) -> Any:
    """Accumulate into a list; a non-list update is wrapped in one."""
    left = list(current) if isinstance(current, (list, tuple)) else (
        [] if current is None else [current]
    )
    right = list(incoming) if isinstance(incoming, (list, tuple)) else [incoming]
    return [*left, *right]


def merge(current: Any, incoming: Any) -> Any:
    """Shallow-merge two mappings."""
    return {**(current or {}), **(incoming or {})}


def keep_first(current: Any, incoming: Any) -> Any:
    """Once set, a value never changes (useful for immutable inputs)."""
    return current if current is not None else incoming


def sum_values(current: Any, incoming: Any) -> Any:
    return (current or 0) + (incoming or 0)


def max_value(current: Any, incoming: Any) -> Any:
    if current is None:
        return incoming
    if incoming is None:
        return current
    return current if current >= incoming else incoming


def apply_update(
    state: Mapping[str, Any],
    update: StateUpdate,
    reducers: Mapping[str, Reducer] | None = None,
) -> dict[str, Any]:
    """Merge a node's partial update into the state.

    没有注册 reducer 的字段直接覆盖；注册了的交给 reducer 合成 ——
    这是「多个节点写同一个字段」时不出错的关键。
    """
    reducers = reducers or {}
    merged = dict(state)
    for key, value in (update or {}).items():
        reducer = reducers.get(key)
        merged[key] = reducer(merged.get(key), value) if reducer else value
    return merged


# ---------------------------------------------------------------- 结果


class GraphStatus(str, Enum):
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    ERROR = "error"


@dataclass(frozen=True)
class GraphResult:
    """Everything a graph run produces: final state plus the audit trail."""

    state: Mapping[str, Any]
    status: GraphStatus
    steps: int
    #: 本次运行实际执行过的节点序列（含循环中的重复访问）
    path: tuple[str, ...] = ()
    trace: tuple[TraceEvent, ...] = ()
    thread_id: str = ""
    error: str | None = None
    #: 原始异常，便于调用方自行处理；不参与序列化
    exception: BaseException | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", frozen_view(self.state))

    @property
    def succeeded(self) -> bool:
        return self.status is GraphStatus.COMPLETED

    def events(self, event_type: str) -> tuple[TraceEvent, ...]:
        return tuple(e for e in self.trace if e.type == event_type)

    def node_durations(self) -> dict[str, float]:
        """Total milliseconds spent inside each node (from trace spans)."""
        totals: dict[str, float] = {}
        for event in self.events(EventType.NODE_START):
            node = event.payload.get("node", "?")
            if event.duration_ms is not None:
                totals[node] = totals.get(node, 0.0) + event.duration_ms
        return totals

    def raise_for_status(self) -> GraphResult:
        if self.succeeded:
            return self
        if self.status is GraphStatus.MAX_STEPS:
            raise GraphStepLimitExceeded(self.steps, self.path)
        raise GraphError(self.error or f"Graph run failed with status {self.status.value}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "steps": self.steps,
            "path": list(self.path),
            "thread_id": self.thread_id,
            "error": self.error,
            "state": dict(self.state),
            "trace": [e.to_dict() for e in self.trace],
        }


# ---------------------------------------------------------------- 定义图


class StateGraph:
    """Builder for a graph of pure functions over a shared state."""

    def __init__(self, reducers: Mapping[str, Reducer] | None = None) -> None:
        self._nodes: dict[str, NodeFn] = {}
        self._edges: dict[str, str] = {}
        self._branches: dict[str, tuple[RouteFn, Mapping[str, str]]] = {}
        self._reducers: dict[str, Reducer] = dict(reducers or {})
        self._entry: str | None = None

    # ------------------------------------------------------------ 构建

    def add_node(self, name: str, fn: NodeFn) -> StateGraph:
        """Register one node. Returns ``self`` so calls can be chained."""
        if not name:
            raise GraphError("Node name must be a non-empty string.")
        if name == END:
            raise GraphError(f"{END!r} is reserved and cannot be a node name.")
        if name in self._nodes:
            raise DuplicateNodeError(name)
        if not callable(fn):
            raise TypeError(f"Node {name!r} must be callable.")
        self._nodes[name] = fn
        return self

    def node(self, name: str) -> Callable[[NodeFn], NodeFn]:
        """Decorator form of :meth:`add_node`.

        >>> graph = StateGraph()
        >>> @graph.node("draft")
        ... def draft(state): return {"draft": "..."}
        """

        def decorator(fn: NodeFn) -> NodeFn:
            self.add_node(name, fn)
            return fn

        return decorator

    def add_edge(self, source: str, target: str) -> StateGraph:
        """Always go from ``source`` to ``target`` (``target`` may be ``END``)."""
        if source == END:
            raise GraphError("END cannot have outgoing edges.")
        if source in self._branches:
            raise GraphError(
                f"Node {source!r} already has a conditional branch; "
                "a node must route either statically or conditionally, not both."
            )
        self._edges[source] = target
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouteFn,
        mapping: Mapping[str, str] | None = None,
    ) -> StateGraph:
        """Route dynamically: ``router(state)`` decides where to go next.

        ``mapping`` 把 router 的返回值翻译成节点名（这样 router 可以返回
        ``"retry"`` / ``"done"`` 这种业务语义，而不是硬编码节点名）。
        不给 mapping 时，router 的返回值就直接当作节点名。
        """
        if source == END:
            raise GraphError("END cannot have outgoing edges.")
        if source in self._edges:
            raise GraphError(
                f"Node {source!r} already has a static edge; "
                "a node must route either statically or conditionally, not both."
            )
        if not callable(router):
            raise TypeError(f"Router for node {source!r} must be callable.")
        self._branches[source] = (router, dict(mapping or {}))
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entry = name
        return self

    def set_finish_point(self, name: str) -> StateGraph:
        """Explicitly end the run after ``name`` (equivalent to an edge to END)."""
        return self.add_edge(name, END)

    # ------------------------------------------------------------ 只读视图

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    @property
    def entry_point(self) -> str | None:
        return self._entry

    @property
    def reducers(self) -> Mapping[str, Reducer]:
        return dict(self._reducers)

    def to_mermaid(self) -> str:
        """Render the graph as a Mermaid flowchart (docs and debugging)."""
        lines = ["graph TD"]
        if self._entry:
            lines.append(f"  __start__ --> {self._entry}")
        for source, target in self._edges.items():
            lines.append(f"  {source} --> {target}")
        for source, (_router, mapping) in self._branches.items():
            if mapping:
                for branch, target in mapping.items():
                    lines.append(f"  {source} -->|{branch}| {target}")
            else:
                lines.append(f"  {source} -->|?| __dynamic__")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"StateGraph(nodes={list(self._nodes)}, entry={self._entry!r})"

    # ------------------------------------------------------------ 校验 + 编译

    def compile(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        max_steps: int = 25,
        name: str = "graph",
    ) -> CompiledGraph:
        """Validate the definition and return a runnable graph.

        校验放在这里而不是 add_* 里，是为了允许「先连边、后加节点」的写法，
        同时保证任何能跑起来的图都是完整的：入口存在、边指向真实节点、
        没有永远走不到的节点。
        """
        if not self._nodes:
            raise GraphError("Cannot compile a graph with no nodes.")
        if self._entry is None:
            raise MissingEntryPointError()
        if self._entry not in self._nodes:
            raise NodeNotFoundError(self._entry, tuple(self._nodes))

        for source, target in self._edges.items():
            self._require_known(source)
            self._require_known(target)
        for source, (_router, mapping) in self._branches.items():
            self._require_known(source)
            for _branch, target in mapping.items():
                self._require_known(target)
        unreachable = self._unreachable_nodes()
        if unreachable:
            raise UnreachableNodeError(unreachable)

        return CompiledGraph(
            name=name,
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            branches=dict(self._branches),
            reducers=dict(self._reducers),
            entry=self._entry,
            checkpointer=checkpointer,
            max_steps=max_steps,
        )

    def _require_known(self, name: str) -> None:
        if name != END and name not in self._nodes:
            raise NodeNotFoundError(name, tuple(self._nodes))

    def _unreachable_nodes(self) -> tuple[str, ...]:
        """BFS from the entry point over both static edges and branch targets."""
        if self._entry is None:  # pragma: no cover - compile 已保证
            return ()
        seen: set[str] = set()
        queue: list[str] = [self._entry]
        while queue:
            node = queue.pop()
            if node == END or node in seen:
                continue
            seen.add(node)
            target = self._edges.get(node)
            if target:
                queue.append(target)
            branch = self._branches.get(node)
            if branch is not None:
                mapping = branch[1]
                # 没有 mapping 的路由是「完全动态」的：静态分析猜不到它会
                # 去哪，因此保守地认为它可以去任何节点（宁可漏报也不误报）
                queue.extend(mapping.values() if mapping else self._nodes)
        return tuple(n for n in self._nodes if n not in seen)


# ---------------------------------------------------------------- 执行图


class CompiledGraph:
    """A validated, runnable graph. Create one via :meth:`StateGraph.compile`."""

    def __init__(
        self,
        *,
        name: str,
        nodes: dict[str, NodeFn],
        edges: dict[str, str],
        branches: dict[str, tuple[RouteFn, Mapping[str, str]]],
        reducers: dict[str, Reducer],
        entry: str,
        checkpointer: Checkpointer | None,
        max_steps: int,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.name = name
        self._nodes = nodes
        self._edges = edges
        self._branches = branches
        self._reducers = reducers
        self._entry = entry
        self._checkpointer = checkpointer
        self.max_steps = max_steps

    # ------------------------------------------------------------ 只读视图

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._nodes)

    @property
    def entry_point(self) -> str:
        return self._entry

    def __repr__(self) -> str:
        return (
            f"CompiledGraph({self.name!r}, nodes={list(self._nodes)}, "
            f"max_steps={self.max_steps})"
        )

    # ------------------------------------------------------------ 运行

    def invoke(
        self,
        initial_state: Mapping[str, Any] | None = None,
        *,
        thread_id: str | None = None,
    ) -> GraphResult:
        """Run the graph to completion from ``initial_state``."""
        return self._run(
            dict(initial_state or {}),
            self._entry,
            thread_id=thread_id or _new_thread_id(),
            start_step=0,
            path=(),
        )

    async def ainvoke(
        self,
        initial_state: Mapping[str, Any] | None = None,
        *,
        thread_id: str | None = None,
    ) -> GraphResult:
        """Async version of :meth:`invoke`; async nodes are awaited in place."""
        return await self._arun(
            dict(initial_state or {}),
            self._entry,
            thread_id=thread_id or _new_thread_id(),
            start_step=0,
            path=(),
        )

    def resume(self, thread_id: str) -> GraphResult:
        """Continue a run from its latest checkpoint.

        典型用法：某一步需要人工确认，进程退出；人点完「通过」之后，
        服务用同一个 ``thread_id`` 把流程接着跑完，前面的节点不会重跑。
        """
        if self._checkpointer is None:
            raise CheckpointError(
                "This graph has no checkpointer; compile(..., checkpointer=...) first."
            )
        snapshot = self._checkpointer.load(thread_id)
        if snapshot is None:
            raise ThreadNotFoundError(thread_id)
        return self._run(
            dict(snapshot.state),
            snapshot.node,
            thread_id=thread_id,
            start_step=snapshot.step,
            path=(),
        )

    async def aresume(self, thread_id: str) -> GraphResult:
        """Async version of :meth:`resume`."""
        if self._checkpointer is None:
            raise CheckpointError(
                "This graph has no checkpointer; compile(..., checkpointer=...) first."
            )
        snapshot = self._checkpointer.load(thread_id)
        if snapshot is None:
            raise ThreadNotFoundError(thread_id)
        return await self._arun(
            dict(snapshot.state),
            snapshot.node,
            thread_id=thread_id,
            start_step=snapshot.step,
            path=(),
        )

    def get_state(self, thread_id: str) -> Checkpoint | None:
        """Latest stored checkpoint for a thread, if any."""
        if self._checkpointer is None:
            return None
        return self._checkpointer.load(thread_id)

    # ------------------------------------------------------------ 内部

    def _run(
        self,
        state: dict[str, Any],
        current: str | None,
        *,
        thread_id: str,
        start_step: int,
        path: tuple[str, ...],
    ) -> GraphResult:
        """Sync driver over the one async core (see ``_arun``)."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self._arun(
                    state,
                    current,
                    thread_id=thread_id,
                    start_step=start_step,
                    path=path,
                )
            )
        raise RuntimeError(
            "invoke() cannot be called from a running event loop; "
            "use `await graph.ainvoke(...)` instead."
        )

    async def _arun(
        self,
        state: dict[str, Any],
        current: str | None,
        *,
        thread_id: str,
        start_step: int,
        path: tuple[str, ...],
    ) -> GraphResult:
        tracer = Tracer()
        steps = start_step
        visited: list[str] = list(path)
        status = GraphStatus.COMPLETED
        error: str | None = None
        exception: BaseException | None = None

        tracer.emit(
            EventType.GRAPH_START,
            step=steps,
            graph=self.name,
            thread_id=thread_id,
            entry=current,
            keys=sorted(state),
        )

        while current is not None and current != END:
            if steps >= self.max_steps:
                status = GraphStatus.MAX_STEPS
                error = f"graph did not finish within max_steps={self.max_steps}"
                tracer.emit(
                    EventType.MAX_STEPS, step=steps, max_steps=self.max_steps
                )
                break

            steps += 1
            node_fn = self._nodes[current]
            try:
                # 节点只拿到只读视图：想直接改状态会当场 TypeError
                with tracer.span(EventType.NODE_START, step=steps, node=current):
                    update = await self._call_node(node_fn, state)
                state = apply_update(state, update, self._reducers)
            except Exception as exc:
                exception = exc
                error = f"{type(exc).__name__}: {exc}"
                status = GraphStatus.ERROR
                tracer.emit(
                    EventType.ERROR, step=steps, node=current, error=error
                )
                break

            visited.append(current)
            tracer.emit(
                EventType.NODE_END,
                step=steps,
                node=current,
                updated=sorted(update or {}),
            )

            target = self._route(current, state, tracer, steps)
            self._checkpoint(tracer, thread_id, steps, target, state, status)
            current = None if target == END else target

        tracer.emit(
            EventType.GRAPH_END,
            step=steps,
            status=status.value,
            path=list(visited),
            keys=sorted(state),
        )

        return GraphResult(
            state=state,
            status=status,
            steps=steps,
            path=tuple(visited),
            trace=tuple(tracer.events),
            thread_id=thread_id,
            error=error,
            exception=exception,
        )

    async def _call_node(self, fn: NodeFn, state: Mapping[str, Any]) -> StateUpdate:
        """Invoke one node, awaiting coroutine nodes.

        同步节点也进线程池：图里挂一个阻塞 IO 的节点（读文件、调一次同步 SDK）
        就够让 ``ainvoke`` 的并发全部失效，而这类节点恰恰最常见。
        """
        view = frozen_view(state)
        if inspect.iscoroutinefunction(fn):
            return await fn(view)
        result = await asyncio.to_thread(fn, view)
        if inspect.isawaitable(result):  # 被装饰器包过的 async 节点
            return await result
        return result

    def _route(
        self,
        source: str,
        state: Mapping[str, Any],
        tracer: Tracer,
        step: int,
    ) -> str:
        """Resolve the next node: conditional branch wins over static edges."""
        branch = self._branches.get(source)
        if branch is not None:
            router, mapping = branch
            key = router(frozen_view(state))
            if inspect.isawaitable(key):
                # router 在同步上下文中被调用，await 不了。与其让 coroutine 对象
                # 变成路由键触发一个莫名其妙的 InvalidRouteError，不如直接说清原因。
                close = getattr(key, "close", None)
                if callable(close):
                    close()
                raise GraphError(
                    f"Branch function for node {source!r} returned an awaitable; "
                    "conditional edges must be synchronous pure functions."
                )
            if mapping and key in mapping:
                target = mapping[key]
            elif key == END or key in self._nodes:
                target = key
            else:
                allowed = tuple(mapping) or (*self._nodes, END)
                raise InvalidRouteError(source, str(key), allowed)
            tracer.emit(
                EventType.ROUTE,
                step=step,
                source=source,
                branch=str(key),
                target=target,
                conditional=True,
            )
            return target

        # 没有出边的节点 = 这条分支到此结束（不必显式连到 END）
        target = self._edges.get(source, END)
        tracer.emit(
            EventType.ROUTE, step=step, source=source, target=target, conditional=False
        )
        return target

    def _checkpoint(
        self,
        tracer: Tracer,
        thread_id: str,
        step: int,
        target: str,
        state: Mapping[str, Any],
        status: GraphStatus,
    ) -> None:
        if self._checkpointer is None:
            return
        snapshot = Checkpoint(
            thread_id=thread_id,
            step=step,
            node=None if target == END else target,
            state=state,
            metadata={"graph": self.name, "status": status.value},
        )
        self._checkpointer.save(snapshot)
        tracer.emit(
            EventType.CHECKPOINT,
            step=step,
            thread_id=thread_id,
            next=snapshot.node,
        )


def _new_thread_id() -> str:
    return f"thread-{uuid.uuid4().hex[:12]}"
