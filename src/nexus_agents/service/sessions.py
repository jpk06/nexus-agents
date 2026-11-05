"""Session state for the HTTP layer.

一个会话是什么
--------------
不是「一条 WebSocket 连接」，而是**一段独立的记忆 + 一把锁**：

* ``memory`` —— 长期记忆（召回 + 写回），让它成为「聊过天」的助手；
* ``lock`` —— 同一会话同时只允许一次运行。两个人（或两个标签页）同时
  往一个会话里发消息，交错的两轮问答会把记忆写成一团乱麻；
* ``messages`` —— 这一会话的完整消息流水，供 ``GET /sessions/{id}`` 审计。

为什么这里是 Protocol + 内存实现
--------------------------------
会话状态最终一定要搬到 Redis 或 Postgres（进程重启就丢的会话不叫会话）。
先在协议后面放一个内存实现，接口对了以后再换存储，路由层一行都不用改。
这也是这一层唯一值得抽象的地方 —— 其余部分没有第二种实现，就别造接口。

TTL 用 monotonic 时钟
---------------------
``time.time()`` 会被 NTP 校正、会被用户手动改，甚至可能往回跳。用系统
时间算过期，就会出现「时钟往回拨一秒，所有会话永不超时」。``monotonic``
只保证一件事：单调递增，这正好是过期判断需要的全部。
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from ..memory import ConversationMemory
from ..types import Message, Role

__all__ = ["Session", "SessionStore", "InMemorySessionStore"]


@dataclass
class Session:
    """One conversation: its memory, its history and its serialisation lock."""

    id: str
    memory: ConversationMemory
    #: 墙上时钟，只用来展示「这个会话是什么时候建的」
    created_at: float
    #: 会话级 system prompt（覆盖 agent 自身的配置）
    system: str | None = None
    #: 完成的轮数。注意：中途断开的那一轮不算
    turns: int = 0
    #: 消息流水（不含 system —— 那是我们注入的，不是用户说的）
    messages: list[Message] = field(default_factory=list)
    #: 同一会话串行执行。``asyncio.Lock`` 在 3.10+ 不绑定事件循环，可以安全地
    #: 在循环外构造。
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    #: monotonic 时间戳，TTL 判断用
    last_used: float = 0.0

    # ------------------------------------------------------------ 状态变更

    def touch(self, now: float | None = None) -> None:
        self.last_used = time.monotonic() if now is None else now

    def record(self, messages: Iterable[Message]) -> None:
        """Append one finished turn to the audit trail.

        system 消息整条跳过：里面有注入的召回片段，存下来既占地方，又会在
        下一次 GET 时让人误以为「用户真说过这些」。
        """
        for message in messages:
            if message.role is Role.SYSTEM:
                continue
            self.messages.append(message)
        self.turns += 1
        self.touch()

    @property
    def busy(self) -> bool:
        return self.lock.locked()

    def recent_history(self, max_messages: int) -> list[Message]:
        """最后若干条消息，但**必须从一条用户消息处下刀**。

        直接 ``messages[-n:]`` 会切在 assistant(tool_calls) 和随之而来的
        tool 结果中间。那半截请求发给模型就是 400：工具调用的“原子单元”
        被拆开了。所以窗口起点往后挪到第一条 user —— 宁可少一轮，不能
        发一个模型必然拒收的请求。
        """
        if max_messages <= 0 or not self.messages:
            return []
        start = max(0, len(self.messages) - max_messages)
        while start < len(self.messages) and self.messages[start].role is not Role.USER:
            start += 1
        return self.messages[start:]


class SessionStore(Protocol):
    """Where sessions live. 换 Redis 时实现这四个方法就够了。"""

    def create(
        self, *, system: str | None = None, memory: ConversationMemory | None = None
    ) -> Session: ...

    def get(self, session_id: str) -> Session | None: ...

    def delete(self, session_id: str) -> bool: ...

    def count(self) -> int: ...


class InMemorySessionStore:
    """Process-local sessions with TTL and LRU eviction.

    为什么要有上限
    --------------
    没有上限的内存会话就是一个缓慢的内存泄漏：每个会话都挂着一份记忆和
    消息流水，只要有人一直建会话，进程迟早 OOM。所以这里同时设了 TTL 和
    容量，用 ``OrderedDict`` 让「最久未使用」的淘汰是 O(1) —— 用普通 dict
    每次淘汰都要全表扫一遍找最小值。

    ``get()`` 会把会话移到队尾（真正的 LRU 语义），也会顺手判掉过期。
    过期只在访问时判断 + 创建时批量清理，不另起后台任务：一个演示服务
    不值得为 GC 多开一个协程。
    """

    def __init__(
        self,
        *,
        max_sessions: int = 1000,
        ttl_seconds: float = 3600.0,
        clock=time.monotonic,
    ) -> None:
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        #: 观测用计数器：被容量挤掉的会话数。生产上这个数应该进监控，
        #: 它一变意味着「有人在刷接口」或者「上限设小了」。
        self.evicted = 0

    # ------------------------------------------------------------ 协议实现

    def create(
        self, *, system: str | None = None, memory: ConversationMemory | None = None
    ) -> Session:
        self._drop_expired()
        while len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)
            self.evicted += 1
        session = Session(
            id=uuid.uuid4().hex,
            memory=memory if memory is not None else ConversationMemory(),
            created_at=time.time(),
            system=system,
            last_used=self._clock(),
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        session = self._sessions.get(session_id)
        if session is None:
            return None
        now = self._clock()
        if now - session.last_used > self.ttl_seconds:
            # 过期即删除：让「拿到的会话一定还活着」成为调用方可以依赖的事实
            del self._sessions[session_id]
            return None
        session.touch(now)
        self._sessions.move_to_end(session_id)
        return session

    def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def count(self) -> int:
        return len(self._sessions)

    # ------------------------------------------------------------ 内部

    def _drop_expired(self) -> None:
        """Clean from the front: 队首就是最久未使用的，过期一定也在前面。

        这样每次创建的开销只和「本次真正过期的数量」有关，而不是和总容量
        有关 —— 满容量时全表扫会让创建退化成 O(n)。
        """
        now = self._clock()
        while self._sessions:
            session_id, session = next(iter(self._sessions.items()))
            if now - session.last_used <= self.ttl_seconds:
                break
            del self._sessions[session_id]
