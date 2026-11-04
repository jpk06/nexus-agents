"""Checkpointing: pause a graph run and continue it later.

为什么需要
----------
多智能体流程经常跑很久 —— 要等人审批、要等外部回调、或者中间某一步
失败了想从那一步重跑。如果状态只存在于内存里，进程一挂就全没了。

这里的做法
----------
把「跑到哪了」抽成一个 :class:`Checkpoint`（线程 id + 步数 + 完整状态 +
下一个节点），存储本身只用一个 :class:`Checkpointer` 协议描述。内存实现
用于测试和单进程；换成 Redis / Postgres 只要实现同一个协议，图的代码
一行都不用改。
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Checkpoint",
    "Checkpointer",
    "InMemoryCheckpointer",
    "frozen_view",
]


def frozen_view(state: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a read-only view of a mapping.

    图把状态交给节点函数时用的就是这个视图：节点想直接改状态会在
    ``TypeError`` 上当场失败，而不是悄悄污染后续节点。
    """
    return MappingProxyType(dict(state))


@dataclass(frozen=True)
class Checkpoint:
    """A durable snapshot of one graph run at one point in time."""

    thread_id: str
    #: 已经执行过的节点数
    step: int
    #: 下一个要执行的节点名；``None`` 表示这条线程已经跑完
    node: str | None
    state: Mapping[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # 状态一旦存进去就不该再被改，所以立刻冻结
        object.__setattr__(self, "state", frozen_view(self.state))
        object.__setattr__(self, "metadata", frozen_view(self.metadata))

    @property
    def finished(self) -> bool:
        return self.node is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "step": self.step,
            "node": self.node,
            "state": dict(self.state),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return f"Checkpoint(thread={self.thread_id!r}, step={self.step}, node={self.node!r})"


@runtime_checkable
class Checkpointer(Protocol):
    """Storage contract for checkpoints. Implement it to persist anywhere."""

    def save(self, checkpoint: Checkpoint) -> None: ...

    def load(self, thread_id: str) -> Checkpoint | None:
        """Return the most recent checkpoint for ``thread_id``, if any."""
        ...

    def history(self, thread_id: str) -> tuple[Checkpoint, ...]:
        """All checkpoints for ``thread_id``, oldest first."""
        ...

    def threads(self) -> tuple[str, ...]: ...

    def clear(self, thread_id: str | None = None) -> None:
        """Drop one thread, or everything when ``thread_id`` is ``None``."""
        ...


class InMemoryCheckpointer:
    """Dict-backed checkpointer: full history per thread, nothing persisted.

    保留**全部**历史而不只是最新一条 —— 排查「状态是在哪一步跑偏的」
    时，能回看每一步的快照价值很大。
    """

    def __init__(self) -> None:
        self._history: dict[str, list[Checkpoint]] = {}

    # ------------------------------------------------------------ Checkpointer

    def save(self, checkpoint: Checkpoint) -> None:
        self._history.setdefault(checkpoint.thread_id, []).append(checkpoint)

    def load(self, thread_id: str) -> Checkpoint | None:
        snapshots = self._history.get(thread_id)
        return snapshots[-1] if snapshots else None

    def history(self, thread_id: str) -> tuple[Checkpoint, ...]:
        return tuple(self._history.get(thread_id, ()))

    def threads(self) -> tuple[str, ...]:
        return tuple(self._history)

    def clear(self, thread_id: str | None = None) -> None:
        if thread_id is None:
            self._history.clear()
        else:
            self._history.pop(thread_id, None)

    # ------------------------------------------------------------ 便利方法

    def at_step(self, thread_id: str, step: int) -> Checkpoint | None:
        """Return the checkpoint recorded after ``step`` node executions."""
        for snapshot in self.history(thread_id):
            if snapshot.step == step:
                return snapshot
        return None

    def __len__(self) -> int:
        return sum(len(v) for v in self._history.values())

    def __contains__(self, thread_id: object) -> bool:
        return thread_id in self._history

    def __repr__(self) -> str:
        return f"InMemoryCheckpointer(threads={len(self._history)}, checkpoints={len(self)})"
