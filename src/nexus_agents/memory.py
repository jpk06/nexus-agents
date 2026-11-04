"""Memory: what the model sees *now*, and what the system remembers *later*.

两个不同的问题，不要混在一起
----------------------------
1. **上下文（context）** —— 这一次请求里到底塞哪些消息。上下文窗口是硬约束，
   塞不下就是 400，或者悄悄变贵、变慢、变糊涂。
2. **长期记忆（memory）** —— 跨会话保存下来的事实，需要时按相关性捞回来。

现实里最常见的做法是把两者都丢掉：要么一直往列表里 append 直到撑爆，要么
一次性把历史全删了。这里各给一个可测的实现。

上下文裁剪的关键细节
--------------------
OpenAI 兼容接口有一条硬规则：``role=tool`` 的消息必须紧跟在带 ``tool_calls``
的 assistant 消息后面，否则请求直接 400。所以裁剪**不能按条数丢**，必须按
「组」丢 —— 一条 assistant(tool_calls) 加上它触发的全部 tool 结果是一个不可
分割的原子单元。:class:`ContextPolicy` 做的就是这件事。

为什么 token 数是估算
---------------------
精确分词要绑死某个模型的分词器（tiktoken 只对 OpenAI 准），而这里的目标是
「别超预算」，不是计费。:class:`HeuristicTokenEstimator` 用「CJK 字符 ≈ 1
token，ASCII ≈ 4 字符 1 token」的启发式，误差 ±15%，零依赖。需要精确值时
实现 :class:`TokenEstimator` 协议换掉即可。

为什么检索不用向量库
--------------------
BM25-lite（idf × 词频饱和 × 长度归一）在几百到几万条记忆上完全够用，零依赖、
零延迟、结果确定 —— 所以测试可以断言排名而不是断言「大概能找到」。
检索接口是协议，将来换 embedding 不用改调用方。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .providers.base import GenerateRequest, LLMProvider
from .types import Message, Role, system_message, user_message

__all__ = [
    "TokenEstimator",
    "HeuristicTokenEstimator",
    "estimate_message_tokens",
    "tokenize",
    "render_transcript",
    "MemoryRecord",
    "MemoryStore",
    "InMemoryMemoryStore",
    "Summarizer",
    "ExtractiveSummarizer",
    "LLMSummarizer",
    "ContextStats",
    "ContextResult",
    "ContextPolicy",
    "ConversationMemory",
    "SUMMARY_PREFIX",
]

#: 摘要消息的固定前缀。写成常量而不是散落在各处，是因为下游可能想识别它。
SUMMARY_PREFIX = "[summary of earlier conversation]"

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+")
_WORD_RE = re.compile(r"[a-z0-9_]+")


def _is_wide(char: str) -> bool:
    """True for characters that a BPE tokenizer usually spends ~1 token on."""
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF  # CJK 扩展 A
        or 0x4E00 <= code <= 0x9FFF  # CJK 基本区
        or 0x3040 <= code <= 0x30FF  # 日文假名
        or 0xAC00 <= code <= 0xD7AF  # 韩文音节
        or 0x3000 <= code <= 0x303F  # CJK 标点
        or 0xFF00 <= code <= 0xFFEF  # 全角字符
    )


# ---------------------------------------------------------------- 估算


@runtime_checkable
class TokenEstimator(Protocol):
    """How many tokens a piece of text will cost.

    三个方法而不是一个，是因为消息有固定开销（role、分隔符、tool_calls 的结构），
    只按 content 长度估算会系统性偏低，最后正好卡在窗口边缘翻车。

    约定：``count_messages(())`` 必须等于**一次请求的固定开销**，裁剪逻辑靠这条
    约定把预算算准。
    """

    def count(self, text: str) -> int: ...

    def count_message(self, message: Message) -> int: ...

    def count_messages(self, messages: Iterable[Message]) -> int: ...


@dataclass(frozen=True)
class HeuristicTokenEstimator:
    """Dependency-free token estimate, tuned for mixed Chinese/English text."""

    #: 平均多少个 ASCII 字符折合一个 token
    chars_per_token: float = 4.0
    #: 每条消息的结构开销（role、分隔符）
    message_overhead: int = 4
    #: 每次请求的固定开销（回复起始标记等）
    request_overhead: int = 3

    def count(self, text: str) -> int:
        if not text:
            return 0
        wide = 0
        for char in text:
            if _is_wide(char):
                wide += 1
        narrow = len(text) - wide
        return max(1, math.ceil(wide + narrow / self.chars_per_token))

    def count_message(self, message: Message) -> int:
        total = self.message_overhead + self.count(message.content)
        if message.name:
            total += self.count(message.name)
        for call in message.tool_calls:
            # 工具调用在线上格式里是 name + JSON 字符串，都算输入
            total += self.count(call.name)
            total += self.count(json.dumps(call.arguments, ensure_ascii=False, sort_keys=True))
        return total

    def count_messages(self, messages: Iterable[Message]) -> int:
        return sum(self.count_message(m) for m in messages) + self.request_overhead


def estimate_message_tokens(
    message: Message, estimator: TokenEstimator | None = None
) -> int:
    """Convenience wrapper so callers don't have to instantiate an estimator."""
    return (estimator or HeuristicTokenEstimator()).count_message(message)


# ---------------------------------------------------------------- 分词


def tokenize(text: str) -> tuple[str, ...]:
    """Split text into retrieval terms.

    中文没有空格，所以不能只按 ``\\w+`` 切：这里把连续的 CJK 串切成字符
    bigram（"景德镇陶瓷" -> 景德 / 德镇 / 镇陶 / 陶瓷）。bigram 不需要词典，
    对中文检索的召回明显好于单字，也不需要引入 jieba。
    """
    lowered = text.lower()
    tokens: list[str] = _WORD_RE.findall(lowered)
    for run in _CJK_RE.findall(lowered):
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tuple(tokens)


def render_transcript(messages: Iterable[Message]) -> str:
    """Flatten messages into readable text (for summaries and debugging)."""
    lines: list[str] = []
    for message in messages:
        label = message.name or message.role.value
        content = message.content.strip()
        if message.tool_calls:
            calls = ", ".join(
                f"{c.name}({json.dumps(c.arguments, ensure_ascii=False)})"
                for c in message.tool_calls
            )
            content = f"{content} -> {calls}".strip()
        if not content:
            continue
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------- 长期记忆


@dataclass(frozen=True)
class MemoryRecord:
    """One fact worth remembering, plus the metadata needed to find it again."""

    text: str
    #: 空字符串 => 用内容哈希自动生成，同样的内容天然去重
    id: str = ""
    tags: frozenset[str] = frozenset()
    created_at: float = field(default_factory=time.time)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("memory text must not be empty")
        if not self.id:
            digest = hashlib.sha256(self.text.strip().encode("utf-8")).hexdigest()
            object.__setattr__(self, "id", digest[:16])
        object.__setattr__(self, "tags", frozenset(self.tags))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "tags": sorted(self.tags),
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
        }

    def __repr__(self) -> str:
        return f"MemoryRecord(id={self.id!r}, tags={sorted(self.tags)}, text={self.text[:40]!r})"


@runtime_checkable
class MemoryStore(Protocol):
    """Storage + retrieval contract. Swap in SQLite/pgvector without touching callers."""

    def add(self, record: MemoryRecord) -> MemoryRecord | None:
        """Store the record, or return ``None`` when an identical one exists."""
        ...

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        tags: Iterable[str] | None = None,
        min_score: float = 0.0,
    ) -> tuple[MemoryRecord, ...]: ...

    def all(self) -> tuple[MemoryRecord, ...]: ...

    def clear(self) -> None: ...

    def __len__(self) -> int:
        """Number of stored records."""
        ...


class InMemoryMemoryStore:
    """Dict-backed memory with BM25-lite ranking.

    排序刻意做成**完全确定**的（分数 -> 时间 -> id），因为「检索结果不稳定」
    会让 agent 的行为不可复现，这类问题极难查。
    """

    #: BM25 的两个自由参数，用标准默认值；没有语料可调时不该乱调
    _K1 = 1.5
    _B = 0.75

    def __init__(
        self,
        *,
        recency_weight: float = 0.0,
        half_life_seconds: float = 7 * 24 * 3600,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if recency_weight < 0:
            raise ValueError("recency_weight must be >= 0")
        if half_life_seconds <= 0:
            raise ValueError("half_life_seconds must be > 0")
        self._records: dict[str, MemoryRecord] = {}
        self._recency_weight = recency_weight
        self._half_life = half_life_seconds
        self._clock = clock

    # ------------------------------------------------------------ 写入

    def add(self, record: MemoryRecord) -> MemoryRecord | None:
        if record.id in self._records:
            return None
        self._records[record.id] = record
        return record

    def extend(self, records: Iterable[MemoryRecord]) -> tuple[MemoryRecord, ...]:
        """Add many records, returning only the ones actually stored."""
        stored = [r for r in (self.add(record) for record in records) if r is not None]
        return tuple(stored)

    def clear(self) -> None:
        self._records.clear()

    # ------------------------------------------------------------ 读取

    def get(self, record_id: str) -> MemoryRecord | None:
        return self._records.get(record_id)

    def all(self) -> tuple[MemoryRecord, ...]:
        return tuple(self._records.values())

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, record_id: object) -> bool:
        return record_id in self._records

    def __repr__(self) -> str:
        return f"InMemoryMemoryStore(records={len(self._records)})"

    # ------------------------------------------------------------ 检索

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        tags: Iterable[str] | None = None,
        min_score: float = 0.0,
    ) -> tuple[MemoryRecord, ...]:
        if k < 1:
            raise ValueError("k must be >= 1")
        query_terms = set(tokenize(query))
        if not query_terms:
            return ()

        required = frozenset(tags) if tags is not None else None
        candidates = [
            r for r in self._records.values() if required is None or required <= r.tags
        ]
        if not candidates:
            return ()

        counts = {r.id: Counter(tokenize(r.text)) for r in candidates}
        lengths = {rid: sum(c.values()) for rid, c in counts.items()}
        total = len(candidates)
        avg_length = sum(lengths.values()) / total or 1.0

        # document frequency 只在候选集上统计：多标签过滤后 idf 才有意义
        df = {
            term: sum(1 for c in counts.values() if term in c) for term in query_terms
        }

        now = self._clock()
        scored: list[tuple[float, MemoryRecord]] = []
        for record in candidates:
            term_counts = counts[record.id]
            length = lengths[record.id] or 1
            score = 0.0
            for term in query_terms:
                tf = term_counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (total - df[term] + 0.5) / (df[term] + 0.5))
                saturation = tf * (self._K1 + 1) / (
                    tf + self._K1 * (1 - self._B + self._B * length / avg_length)
                )
                score += idf * saturation
            if score <= 0:
                continue
            if self._recency_weight:
                age = max(now - record.created_at, 0.0)
                score += self._recency_weight * 2 ** (-age / self._half_life)
            if score < min_score:
                continue
            scored.append((score, record))

        scored.sort(key=lambda item: (-item[0], -item[1].created_at, item[1].id))
        return tuple(record for _, record in scored[:k])


# ---------------------------------------------------------------- 摘要


@runtime_checkable
class Summarizer(Protocol):
    """Compress a chunk of conversation into a short paragraph."""

    def summarize(self, messages: tuple[Message, ...], *, max_chars: int = 600) -> str: ...


@dataclass(frozen=True)
class ExtractiveSummarizer:
    """Offline summarizer: keep the gist of each message, drop the bulk.

    没有 API Key 也要能压缩上下文 —— 否则「记得住」这件事在测试和示例里就
    永远是假的。策略是从后往前取，优先保留人和助手的发言，工具输出只用来补位。
    """

    max_lines: int = 8
    line_chars: int = 120

    def summarize(self, messages: tuple[Message, ...], *, max_chars: int = 600) -> str:
        entries: list[tuple[int, str, bool]] = []
        for index, message in enumerate(messages):
            text = message.content.strip()
            if not text:
                continue
            first_line = text.splitlines()[0].strip()
            label = message.name or message.role.value
            line = f"- {label}: {first_line[: self.line_chars]}"
            entries.append((index, line, message.role is Role.TOOL))

        if not entries:
            return ""

        # 倒序挑：越靠后的越接近当前话题
        chosen: list[int] = []
        budget = self.max_lines
        for index, _, is_tool in reversed(entries):
            if is_tool:
                continue
            if budget <= 0:
                break
            chosen.append(index)
            budget -= 1
        for index, _, is_tool in reversed(entries):
            if budget <= 0:
                break
            if not is_tool:
                continue
            chosen.append(index)
            budget -= 1

        keep = set(chosen)
        text = "\n".join(line for index, line, _ in entries if index in keep)
        return text[:max_chars].rstrip()


class LLMSummarizer:
    """Summarize with a real model, degrade to extractive when it fails.

    摘要失败（超时、限流、返回空）不该让整次对话崩掉 —— 这不是「错误」，只是
    「这次没压缩成」。fallback 让降级路径也是一条被测试过的路径。
    """

    DEFAULT_PROMPT = (
        "You compress conversation history for another assistant. "
        "Keep decisions, constraints, numbers, names and open questions. "
        "Drop pleasantries and redundant tool output. Reply with plain text only."
    )

    def __init__(
        self,
        provider: LLMProvider,
        *,
        system_prompt: str | None = None,
        fallback: Summarizer | None = None,
        max_tokens: int | None = 400,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt or self.DEFAULT_PROMPT
        self.fallback = fallback if fallback is not None else ExtractiveSummarizer()
        self.max_tokens = max_tokens

    def summarize(self, messages: tuple[Message, ...], *, max_chars: int = 600) -> str:
        transcript = render_transcript(messages)
        if not transcript:
            return ""
        request = GenerateRequest(
            messages=(system_message(self.system_prompt), user_message(transcript)),
            max_tokens=self.max_tokens,
        )
        try:
            response = self.provider.generate(request)
        except Exception:  # 摘要失败必须降级，不能中断主流程
            return self.fallback.summarize(messages, max_chars=max_chars)
        text = response.message.content.strip()
        if not text:
            return self.fallback.summarize(messages, max_chars=max_chars)
        return text[:max_chars]


# ---------------------------------------------------------------- 上下文裁剪


@dataclass(frozen=True)
class ContextStats:
    """What the policy did. 可断言、可打日志 —— 裁剪必须是可观测的。"""

    tokens_before: int
    tokens_after: int
    #: 发出去的消息条数（含钉住的消息与摘要）
    kept: int
    #: 被裁掉的消息条数
    dropped: int
    had_summary: bool
    #: 剪完仍然超预算（当前请求本身就太大了）
    truncated: bool = False

    @property
    def saved(self) -> int:
        return max(self.tokens_before - self.tokens_after, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "kept": self.kept,
            "dropped": self.dropped,
            "had_summary": self.had_summary,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class ContextResult:
    messages: tuple[Message, ...]
    stats: ContextStats


@dataclass(frozen=True)
class _Unit:
    """裁剪的最小单位：不可再分的消息序列。"""

    messages: tuple[Message, ...]
    pinned: bool
    tokens: int


@dataclass(frozen=True)
class ContextPolicy:
    """Fit a conversation into a token budget without breaking tool pairing."""

    max_tokens: int = 4000
    estimator: TokenEstimator = field(default_factory=HeuristicTokenEstimator)
    #: 第一条用户消息通常就是任务本身，丢掉它模型会彻底跑偏
    keep_first_user: bool = True
    #: 给了摘要器才做摘要，否则只做前缀裁剪
    summarizer: Summarizer | None = None
    summary_max_chars: int = 600

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.summary_max_chars < 1:
            raise ValueError("summary_max_chars must be >= 1")

    # ------------------------------------------------------------ 公共接口

    def apply(self, messages: Sequence[Message]) -> ContextResult:
        if not messages:
            raise ValueError("cannot build a context from an empty message list")

        units = self._units(messages)
        pinned = [u for u in units if u.pinned]
        movable = [u for u in units if not u.pinned]
        pinned_tokens = sum(u.tokens for u in pinned)
        reserve = self.estimator.count_messages(())
        # 开了摘要就先切出 1/4 预算留给它：否则贪心填充会把空间吃干净，
        # 摘要永远「差一点放不进去」，功能等于没写
        summary_room = self.max_tokens // 4 if self.summarizer is not None else 0
        budget = max(self.max_tokens - pinned_tokens - reserve - summary_room, 0)

        # 从最新往回装：超出预算就整段丢掉更老的，保证留下的是连续的后缀
        kept: list[_Unit] = []
        used = 0
        for unit in reversed(movable):
            if used + unit.tokens <= budget:
                kept.append(unit)
                used += unit.tokens
                continue
            if not kept:
                # 最新的请求自己就超预算：宁可超，也不能把当前问题扔掉
                kept.append(unit)
                used += unit.tokens
            break
        kept.reverse()

        # 兜底一：输入本身可能是畸形的（开头就是孤立的 tool 结果），
        # 这种消息发出去一定 400，直接丢掉
        while kept and kept[0].messages[0].role is Role.TOOL:
            used -= kept.pop(0).tokens

        # 兜底二：真的发生过裁剪时，窗口可能是从「一句没头没尾的 assistant
        # 回复」开始的 —— 它回答的那个问题已经被裁掉了，留着只让模型困惑。
        # 不带 tool_calls 的裸回复才可以丢：带工具调用说明后面还挂着结果，
        # 那是有用的状态，不能一起扔。至少留一个单元，别把上下文裁空。
        if len(kept) < len(movable):
            while (
                len(kept) > 1
                and kept[0].messages[0].role is Role.ASSISTANT
                and not kept[0].messages[0].tool_calls
            ):
                used -= kept.pop(0).tokens

        dropped = movable[: len(movable) - len(kept)]
        summary = self._build_summary(dropped, budget - used + summary_room)

        systems = [m for u in pinned if u.messages[0].role is Role.SYSTEM for m in u.messages]
        others = [m for u in pinned if u.messages[0].role is not Role.SYSTEM for m in u.messages]

        out: list[Message] = list(systems)
        if summary is not None:
            out.append(summary)
        out.extend(others)
        for unit in kept:
            out.extend(unit.messages)

        after = self.estimator.count_messages(out)
        stats = ContextStats(
            tokens_before=self.estimator.count_messages(messages),
            tokens_after=after,
            kept=len(out),
            dropped=sum(len(u.messages) for u in dropped),
            had_summary=summary is not None,
            truncated=after > self.max_tokens,
        )
        return ContextResult(messages=tuple(out), stats=stats)

    def trim(self, messages: Sequence[Message]) -> tuple[Message, ...]:
        """Only the messages, when the caller doesn't care about the stats."""
        return self.apply(messages).messages

    # ------------------------------------------------------------ 内部

    def _unit(self, messages: tuple[Message, ...], *, pinned: bool) -> _Unit:
        # 只算消息本身的开销：request_overhead 是每次请求一份，不属于任何一个单元
        return _Unit(
            messages=messages,
            pinned=pinned,
            tokens=sum(self.estimator.count_message(m) for m in messages),
        )

    def _units(self, messages: Sequence[Message]) -> list[_Unit]:
        units: list[_Unit] = []
        index = 0
        first_user_seen = False
        while index < len(messages):
            message = messages[index]
            if message.role is Role.SYSTEM:
                units.append(self._unit((message,), pinned=True))
                index += 1
                continue
            if message.role is Role.ASSISTANT and message.tool_calls:
                # assistant(tool_calls) + 它的全部 tool 结果 = 原子单元
                group = [message]
                cursor = index + 1
                while cursor < len(messages) and messages[cursor].role is Role.TOOL:
                    group.append(messages[cursor])
                    cursor += 1
                units.append(self._unit(tuple(group), pinned=False))
                index = cursor
                continue
            pinned = message.role is Role.USER and self.keep_first_user and not first_user_seen
            if message.role is Role.USER:
                first_user_seen = True
            units.append(self._unit((message,), pinned=pinned))
            index += 1
        return units

    def _build_summary(self, dropped: list[_Unit], remaining: int) -> Message | None:
        if not dropped or self.summarizer is None:
            return None
        text = self.summarizer.summarize(
            tuple(m for u in dropped for m in u.messages), max_chars=self.summary_max_chars
        ).strip()
        if not text:
            return None

        overhead = self.estimator.count_message(system_message(SUMMARY_PREFIX))
        room = max(remaining, 0) - overhead
        if room < 8:
            # 剩余空间连摘要标题都放不下，硬塞只会挤掉真正有用的近况
            return None
        text = self._fit(text, room)
        return system_message(f"{SUMMARY_PREFIX}\n{text}")

    def _fit(self, text: str, room: int) -> str:
        """把摘要压进 ``room`` 个 token 里（估算器和实际分词总有偏差，按比例收敛）。"""
        for _ in range(10):
            current = self.estimator.count(text)
            if current <= room:
                break
            cut = max(1, min(int(len(text) * room / current * 0.95), len(text) - 1))
            text = text[:cut].rstrip()
        return text


# ---------------------------------------------------------------- 门面


class ConversationMemory:
    """Store + recall + context in one object, so the agent sees one dependency.

    ``agent.run(..., memory=memory)`` 会用 ``prime()`` 拼出带召回结果的初始消息，
    跑完再把这一轮存回去。想换成 Redis / 向量库，只要实现 :class:`MemoryStore`。
    """

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        policy: ContextPolicy | None = None,
    ) -> None:
        self.store: MemoryStore = store if store is not None else InMemoryMemoryStore()
        self.policy = policy if policy is not None else ContextPolicy()

    # ------------------------------------------------------------ 写入

    def remember(
        self,
        text: str,
        *,
        tags: Iterable[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> MemoryRecord | None:
        return self.store.add(
            MemoryRecord(text=text, tags=frozenset(tags), metadata=dict(metadata or {}))
        )

    def remember_turn(
        self, question: str, answer: str, *, tags: Iterable[str] = ()
    ) -> tuple[MemoryRecord, ...]:
        """Store one Q/A pair as a single record —— 分开存会丢掉问答对应关系。"""
        if not answer.strip():
            return ()
        record = self.remember(f"Q: {question.strip()}\nA: {answer.strip()}", tags=tags)
        return (record,) if record is not None else ()

    def remember_messages(
        self,
        messages: Iterable[Message],
        *,
        tags: Iterable[str] = (),
        include_tool_output: bool = False,
    ) -> int:
        """Bulk-store conversational messages. Returns how many were new."""
        added = 0
        for message in messages:
            if not message.content.strip():
                continue
            if message.role is Role.TOOL and not include_tool_output:
                continue
            if message.role is Role.SYSTEM:
                continue
            record = self.remember(message.content, tags=(*tags, message.role.value))
            if record is not None:
                added += 1
        return added

    # ------------------------------------------------------------ 读取

    def recall(
        self,
        query: str,
        *,
        k: int = 5,
        tags: Iterable[str] | None = None,
        min_score: float = 0.0,
    ) -> tuple[MemoryRecord, ...]:
        return self.store.search(query, k=k, tags=tags, min_score=min_score)

    def recall_messages(
        self,
        query: str,
        *,
        k: int = 5,
        max_chars: int = 400,
        header: str = "Relevant memory from earlier sessions:",
    ) -> tuple[Message, ...]:
        """Recalled facts as one system message, or nothing when there is no hit."""
        hits = self.recall(query, k=k)
        if not hits:
            return ()
        lines = []
        for record in hits:
            snippet = record.text.strip().replace("\n", " ")
            if len(snippet) > max_chars:
                snippet = snippet[:max_chars].rstrip() + "..."
            lines.append(f"- {snippet}")
        return (system_message(f"{header}\n" + "\n".join(lines)),)

    # ------------------------------------------------------------ 上下文

    def context(self, messages: Sequence[Message]) -> ContextResult:
        return self.policy.apply(messages)

    def prime(
        self,
        user_input: str,
        *,
        system: str | None = None,
        recall: bool = True,
        k: int = 5,
    ) -> tuple[Message, ...]:
        """Assemble the opening messages for a run, recall included."""
        messages: list[Message] = []
        if system:
            messages.append(system_message(system))
        if recall:
            messages.extend(self.recall_messages(user_input, k=k))
        messages.append(user_message(user_input))
        return tuple(messages)

    def __repr__(self) -> str:
        return f"ConversationMemory(records={len(self.store)}, max_tokens={self.policy.max_tokens})"
