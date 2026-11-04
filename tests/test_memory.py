"""Tests for the memory layer: estimation, retrieval, context trimming, digestion."""

from __future__ import annotations

import pytest

from conftest import add
from nexus_agents import (
    SUMMARY_PREFIX,
    AgentConfig,
    ContextPolicy,
    ConversationMemory,
    ExtractiveSummarizer,
    HeuristicTokenEstimator,
    InMemoryMemoryStore,
    LLMSummarizer,
    MemoryRecord,
    ReActAgent,
    estimate_message_tokens,
    tool,
)
from nexus_agents.memory import MemoryStore, Summarizer, TokenEstimator, render_transcript, tokenize
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.tracing import EventType
from nexus_agents.types import (
    Message,
    Role,
    ToolCall,
    assistant_message,
    system_message,
    user_message,
)

# ---------------------------------------------------------------- 测试替身


class FlatEstimator:
    """每条消息恒为 1 token（另加 1 token 请求开销）—— 让预算测试可以精确算账。"""

    def count(self, text: str) -> int:
        return 1

    def count_message(self, message: Message) -> int:
        return 1

    def count_messages(self, messages) -> int:
        return sum(self.count_message(m) for m in messages) + 1


class FixedSummarizer:
    def __init__(self, text: str = "earlier: user asked about ceramics") -> None:
        self.text = text
        self.calls: list[tuple[Message, ...]] = []

    def summarize(self, messages: tuple[Message, ...], *, max_chars: int = 600) -> str:
        self.calls.append(messages)
        return self.text[:max_chars]


class BoomProvider:
    def generate(self, request):
        raise RuntimeError("upstream exploded")


@tool
def big() -> str:
    """Return a large blob of text."""
    return "x" * 400


def group(call_id: str = "c1") -> list[Message]:
    """一条 assistant（并行两次调用）+ 两条 tool 结果，共 3 条消息的原子单元。"""
    second = call_id + "b"
    return [
        assistant_message(
            tool_calls=(
                ToolCall(id=call_id, name="big", arguments={}),
                ToolCall(id=second, name="big", arguments={}),
            )
        ),
        Message(role=Role.TOOL, tool_call_id=call_id, name="big", content="ok"),
        Message(role=Role.TOOL, tool_call_id=second, name="big", content="ok"),
    ]


def conversation(turns: int, *, size: int = 12) -> list[Message]:
    messages: list[Message] = []
    for i in range(turns):
        messages.append(user_message(f"question {i} " + "x" * size))
        messages.append(assistant_message(f"answer {i} " + "y" * size))
    return messages


def assert_tool_pairing_is_valid(messages: tuple[Message, ...]) -> None:
    """任何 tool 消息前面都必须有一条带对应 tool_call_id 的 assistant。"""
    available: set[str] = set()
    for message in messages:
        if message.role is Role.ASSISTANT:
            available = {c.id for c in message.tool_calls}
        elif message.role is Role.TOOL:
            assert message.tool_call_id in available, "orphan tool message in context"


# ---------------------------------------------------------------- Token 估算


def test_empty_text_costs_nothing():
    assert HeuristicTokenEstimator().count("") == 0


def test_ascii_text_is_roughly_four_chars_per_token():
    estimator = HeuristicTokenEstimator()

    assert estimator.count("abcdefgh") == 2
    assert estimator.count("abc") == 1


def test_cjk_chars_cost_one_token_each():
    assert HeuristicTokenEstimator().count("你好世界") == 4


def test_mixed_text_adds_both_halves():
    assert HeuristicTokenEstimator().count("a你好") == 3


def test_message_cost_includes_structural_overhead():
    estimator = HeuristicTokenEstimator()

    assert estimator.count_message(assistant_message("hello")) == 6


def test_tool_calls_are_counted_not_ignored():
    estimator = HeuristicTokenEstimator()
    plain = assistant_message("")
    calling = assistant_message(tool_calls=(ToolCall(id="c", name="add", arguments={"a": 1}),))

    assert estimator.count_message(calling) > estimator.count_message(plain)


def test_message_name_is_counted():
    estimator = HeuristicTokenEstimator()
    named = Message(role=Role.ASSISTANT, content="hi", name="researcher")

    assert estimator.count_message(named) > estimator.count_message(assistant_message("hi"))


def test_count_messages_adds_request_overhead():
    estimator = HeuristicTokenEstimator()
    messages = [user_message("hello")]

    assert estimator.count_messages(messages) == (
        estimator.count_message(messages[0]) + estimator.request_overhead
    )


def test_estimate_message_tokens_wrapper_uses_the_default_estimator():
    assert estimate_message_tokens(user_message("hello")) == 4 + 2


def test_estimate_message_tokens_accepts_a_custom_estimator():
    assert estimate_message_tokens(user_message("hello"), FlatEstimator()) == 1


def test_heuristic_estimator_satisfies_the_protocol():
    assert isinstance(HeuristicTokenEstimator(), TokenEstimator)


def test_custom_estimator_is_honoured_by_the_policy():
    policy = ContextPolicy(max_tokens=10, estimator=FlatEstimator())

    assert policy.estimator.count("anything at all") == 1


# ---------------------------------------------------------------- 分词与渲染


def test_tokenize_lowercases_latin_words():
    assert tokenize("Hello World") == ("hello", "world")


def test_tokenize_splits_cjk_into_bigrams():
    assert tokenize("陶瓷贸易") == ("陶瓷", "瓷贸", "贸易")


def test_tokenize_keeps_single_cjk_char():
    assert tokenize("陶") == ("陶",)


def test_tokenize_handles_mixed_scripts():
    assert tokenize("AI 陶瓷") == ("ai", "陶瓷")


def test_tokenize_ignores_punctuation():
    assert tokenize("hello, world!") == ("hello", "world")


def test_tokenize_of_empty_text_is_empty():
    assert tokenize("   ") == ()


def test_render_transcript_labels_every_message():
    text = render_transcript([user_message("hi"), assistant_message("hello")])

    assert "user: hi" in text
    assert "assistant: hello" in text


def test_render_transcript_renders_tool_calls():
    message = assistant_message(tool_calls=(ToolCall(id="c", name="add", arguments={"a": 1}),))

    assert "add(" in render_transcript([message])


def test_render_transcript_skips_empty_content():
    assert render_transcript([assistant_message("")]) == ""


# ---------------------------------------------------------------- MemoryRecord


def test_record_rejects_blank_text():
    with pytest.raises(ValueError):
        MemoryRecord(text="   ")


def test_record_derives_a_stable_id_from_content():
    first = MemoryRecord(text="the sky is blue")
    second = MemoryRecord(text="the sky is blue")

    assert first.id == second.id
    assert len(first.id) == 16


def test_record_id_differs_for_different_text():
    assert MemoryRecord(text="a").id != MemoryRecord(text="b").id


def test_record_keeps_an_explicit_id():
    assert MemoryRecord(text="a", id="custom").id == "custom"


def test_record_normalises_tags_to_a_frozenset():
    record = MemoryRecord(text="a", tags=["x", "x", "y"])

    assert record.tags == frozenset({"x", "y"})


def test_record_metadata_is_read_only():
    record = MemoryRecord(text="a", metadata={"k": 1})

    with pytest.raises(TypeError):
        record.metadata["k"] = 2  # type: ignore[index]


def test_record_round_trips_through_a_dict():
    record = MemoryRecord(text="a", tags=["x"], metadata={"k": 1})

    data = record.to_dict()

    assert data["text"] == "a"
    assert data["tags"] == ["x"]
    assert data["metadata"] == {"k": 1}


def test_record_repr_is_readable():
    assert "abc" in repr(MemoryRecord(text="abc"))


# ---------------------------------------------------------------- InMemoryMemoryStore


@pytest.fixture
def store() -> InMemoryMemoryStore:
    return InMemoryMemoryStore()


def test_store_adds_and_counts(store):
    record = store.add(MemoryRecord(text="ceramics are fired at 1300C"))

    assert record is not None
    assert len(store) == 1


def test_store_deduplicates_identical_content(store):
    store.add(MemoryRecord(text="same"))
    duplicate = store.add(MemoryRecord(text="same"))

    assert duplicate is None
    assert len(store) == 1


def test_store_returns_new_records_only_when_extending(store):
    stored = store.extend([MemoryRecord(text="a"), MemoryRecord(text="a"), MemoryRecord(text="b")])

    assert len(stored) == 2
    assert len(store) == 2


def test_store_get_by_id(store):
    record = store.add(MemoryRecord(text="a"))
    assert record is not None

    assert store.get(record.id) == record
    assert store.get("missing") is None


def test_store_all_preserves_insertion_order(store):
    for text in ("a", "b", "c"):
        store.add(MemoryRecord(text=text))

    assert [r.text for r in store.all()] == ["a", "b", "c"]


def test_store_clear(store):
    store.add(MemoryRecord(text="a"))
    store.clear()

    assert len(store) == 0


def test_store_contains_checks_ids(store):
    record = store.add(MemoryRecord(text="a"))
    assert record is not None

    assert record.id in store
    assert "nope" not in store


def test_store_repr_counts_records(store):
    store.add(MemoryRecord(text="a"))

    assert "1" in repr(store)


def test_store_search_returns_overlapping_records(store):
    store.add(MemoryRecord(text="the kiln temperature is 1300C"))
    store.add(MemoryRecord(text="glaze chemistry is hard"))

    hits = store.search("kiln temperature")

    assert len(hits) == 1
    assert "kiln" in hits[0].text


def test_store_search_ranks_more_relevant_first(store):
    store.add(MemoryRecord(text="kiln temperature"))
    store.add(MemoryRecord(text="kiln temperature and glaze and cobalt"))

    hits = store.search("kiln temperature glaze cobalt")

    assert len(hits) == 2
    assert "cobalt" in hits[0].text


def test_store_search_returns_nothing_for_unknown_terms(store):
    store.add(MemoryRecord(text="kiln"))

    assert store.search("blockchain") == ()


def test_store_search_of_blank_query_is_empty(store):
    store.add(MemoryRecord(text="kiln"))

    assert store.search("   ") == ()


def test_store_search_respects_k(store):
    for i in range(5):
        store.add(MemoryRecord(text=f"kiln note {i}"))

    assert len(store.search("kiln", k=2)) == 2


def test_store_search_rejects_zero_k(store):
    with pytest.raises(ValueError):
        store.search("kiln", k=0)


def test_store_search_filters_by_tags(store):
    store.add(MemoryRecord(text="kiln note", tags=["ceramics"]))
    store.add(MemoryRecord(text="kiln note other", tags=["misc"]))

    hits = store.search("kiln", tags=["ceramics"])

    assert len(hits) == 1
    assert hits[0].tags == frozenset({"ceramics"})


def test_store_search_requires_all_requested_tags(store):
    store.add(MemoryRecord(text="kiln note", tags=["ceramics", "draft"]))

    assert store.search("kiln", tags=["ceramics", "final"]) == ()


def test_store_search_matches_chinese_queries(store):
    store.add(MemoryRecord(text="景德镇陶瓷市场规模在增长"))

    hits = store.search("景德镇")

    assert len(hits) == 1


def test_store_search_honours_min_score(store):
    store.add(MemoryRecord(text="kiln"))

    assert store.search("kiln", min_score=99.0) == ()


def test_store_search_is_deterministic(store):
    for i in range(5):
        store.add(MemoryRecord(text=f"kiln note {i}"))

    first = [r.id for r in store.search("kiln note", k=5)]
    second = [r.id for r in store.search("kiln note", k=5)]

    assert first == second


def test_store_recency_weight_prefers_newer_records():
    store = InMemoryMemoryStore(recency_weight=1.0, half_life_seconds=10, clock=lambda: 100.0)
    store.add(MemoryRecord(text="kiln alpha", created_at=0.0))
    store.add(MemoryRecord(text="kiln alpha beta", created_at=100.0))

    hits = store.search("kiln")

    assert hits[0].created_at == 100.0


def test_store_rejects_negative_recency_weight():
    with pytest.raises(ValueError):
        InMemoryMemoryStore(recency_weight=-1)


def test_store_rejects_non_positive_half_life():
    with pytest.raises(ValueError):
        InMemoryMemoryStore(half_life_seconds=0)


def test_store_satisfies_the_protocol(store):
    assert isinstance(store, MemoryStore)


# ---------------------------------------------------------------- 摘要器


def test_extractive_summarizer_keeps_one_line_per_message():
    summary = ExtractiveSummarizer().summarize(
        (user_message("first line\nsecond line"), assistant_message("reply"))
    )

    assert "first line" in summary
    assert "second line" not in summary


def test_extractive_summarizer_skips_empty_messages():
    assert ExtractiveSummarizer().summarize((assistant_message(""),)) == ""


def test_extractive_summarizer_truncates_to_max_chars():
    summary = ExtractiveSummarizer().summarize((user_message("x" * 500),), max_chars=40)

    assert len(summary) <= 40


def test_extractive_summarizer_limits_lines():
    messages = tuple(user_message(f"line {i}") for i in range(20))

    summary = ExtractiveSummarizer(max_lines=3).summarize(messages)

    assert len(summary.splitlines()) == 3


def test_extractive_summarizer_keeps_recent_lines():
    messages = tuple(user_message(f"line {i}") for i in range(10))

    summary = ExtractiveSummarizer(max_lines=2).summarize(messages)

    assert "line 9" in summary


def test_extractive_summarizer_prefers_speech_over_tool_output():
    messages = (
        user_message("important question"),
        Message(role=Role.TOOL, name="search", content="verbose tool output"),
    )

    summary = ExtractiveSummarizer(max_lines=1).summarize(messages)

    assert "important question" in summary


def test_extractive_summarizer_fills_leftover_slots_with_tool_output():
    messages = (
        user_message("question"),
        Message(role=Role.TOOL, name="search", content="tool finding"),
        Message(role=Role.TOOL, name="search", content="more finding"),
    )

    summary = ExtractiveSummarizer(max_lines=3).summarize(messages)

    assert "tool finding" in summary
    assert "more finding" in summary


def test_extractive_summarizer_satisfies_the_protocol():
    assert isinstance(ExtractiveSummarizer(), Summarizer)


def test_llm_summarizer_returns_model_text():
    provider = ScriptedProvider(["condensed history"])
    summarizer = LLMSummarizer(provider)

    assert summarizer.summarize((user_message("long chat"),)) == "condensed history"


def test_llm_summarizer_sends_a_system_prompt_and_transcript():
    provider = ScriptedProvider(["ok"])

    LLMSummarizer(provider, system_prompt="compress please").summarize((user_message("chat"),))

    request = provider.last_request
    assert request is not None
    assert request.messages[0].content == "compress please"
    assert "chat" in request.messages[1].content


def test_llm_summarizer_truncates_long_answers():
    summarizer = LLMSummarizer(ScriptedProvider(["y" * 200]))

    assert len(summarizer.summarize((user_message("chat"),), max_chars=20)) == 20


def test_llm_summarizer_falls_back_on_empty_answer():
    summarizer = LLMSummarizer(ScriptedProvider([""]))

    assert "chat" in summarizer.summarize((user_message("chat"),))


def test_llm_summarizer_falls_back_when_the_provider_fails():
    summarizer = LLMSummarizer(BoomProvider())

    assert "chat" in summarizer.summarize((user_message("chat"),))


def test_llm_summarizer_returns_empty_for_empty_transcript():
    assert LLMSummarizer(ScriptedProvider(["x"])).summarize((assistant_message(""),)) == ""


# ---------------------------------------------------------------- ContextPolicy


def test_policy_leaves_small_conversations_untouched():
    messages = conversation(2)
    result = ContextPolicy(max_tokens=4000).apply(messages)

    assert list(result.messages) == messages
    assert result.stats.dropped == 0
    assert result.stats.had_summary is False


def test_policy_drops_oldest_messages_when_over_budget():
    result = ContextPolicy(max_tokens=60).apply(conversation(10))

    assert result.stats.dropped > 0
    assert result.stats.tokens_after <= 60
    assert "question 9" in result.messages[-2].content


def test_policy_pins_the_system_prompt():
    messages = [system_message("you are terse"), *conversation(10)]

    result = ContextPolicy(max_tokens=40).apply(messages)

    assert result.messages[0].content == "you are terse"


def test_policy_pins_the_first_user_message_by_default():
    result = ContextPolicy(max_tokens=40).apply(conversation(10))

    assert result.messages[0].role is Role.USER
    assert "question 0" in result.messages[0].content


def test_policy_can_drop_the_first_user_message():
    result = ContextPolicy(max_tokens=40, keep_first_user=False).apply(conversation(10))

    assert all("question 0 " not in m.content for m in result.messages)


def test_policy_never_splits_a_tool_call_from_its_results():
    messages: list[Message] = [
        system_message("sys"),
        user_message("task"),
        *group(),
        user_message("next"),
    ]
    result = ContextPolicy(max_tokens=4, estimator=FlatEstimator()).apply(messages)

    # 预算只够最新的那条 user，整个工具单元被完整丢弃
    assert [m.role.value for m in result.messages] == ["system", "user", "user"]
    assert_tool_pairing_is_valid(result.messages)


def test_policy_keeps_a_whole_tool_group_when_it_fits():
    messages: list[Message] = [
        system_message("sys"),
        user_message("task"),
        *group(),
        user_message("next"),
    ]
    result = ContextPolicy(max_tokens=7, estimator=FlatEstimator()).apply(messages)

    assert [m.role.value for m in result.messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "user",
    ]
    assert_tool_pairing_is_valid(result.messages)


def test_policy_drops_a_leading_orphan_tool_message():
    messages: list[Message] = [
        Message(role=Role.TOOL, tool_call_id="ghost", name="x", content="stale"),
        user_message("hi"),
    ]

    result = ContextPolicy(max_tokens=200).apply(messages)

    assert [m.role.value for m in result.messages] == ["user"]


def test_policy_does_not_start_the_window_with_a_dangling_reply():
    messages = conversation(10)

    result = ContextPolicy(max_tokens=60).apply(messages)

    # 第一条（钉住的）user 之后，窗口必须从一条 user 消息开始
    assert result.messages[1].role is Role.USER


def test_policy_keeps_a_tool_group_at_the_head_when_it_carries_results():
    messages: list[Message] = [
        system_message("sys"),
        user_message("task"),
        *group(),
    ]
    result = ContextPolicy(max_tokens=4, estimator=FlatEstimator()).apply(messages)

    roles = [m.role.value for m in result.messages]
    assert roles == ["system", "user", "assistant", "tool", "tool"]


def test_policy_keeps_an_oversized_latest_message_and_flags_it():
    result = ContextPolicy(max_tokens=40).apply([user_message("x" * 800)])

    assert len(result.messages) == 1
    assert result.stats.truncated is True


def test_policy_adds_a_summary_when_configured():
    summarizer = FixedSummarizer()
    policy = ContextPolicy(max_tokens=120, summarizer=summarizer, summary_max_chars=200)

    result = policy.apply(conversation(10))

    assert result.stats.had_summary is True
    assert result.messages[0].content.startswith(SUMMARY_PREFIX)
    assert summarizer.calls, "summarizer was never asked to compress anything"


def test_policy_summary_only_covers_dropped_messages():
    summarizer = FixedSummarizer()
    policy = ContextPolicy(max_tokens=120, summarizer=summarizer)

    result = policy.apply(conversation(10))
    summarized_text = "\n".join(m.content for m in summarizer.calls[0])

    # 第一条 user 消息是被钉住的，不在摘要范围内
    assert "answer 0" in summarized_text
    assert "question 9" not in summarized_text
    assert result.stats.dropped == len(summarizer.calls[0])


def test_policy_shrinks_an_oversized_summary_to_fit():
    policy = ContextPolicy(max_tokens=120, summarizer=FixedSummarizer(text="z" * 3000))

    result = policy.apply(conversation(10))

    assert result.stats.had_summary is True
    assert result.stats.tokens_after <= 120


def test_policy_skips_summary_when_there_is_no_room():
    summarizer = FixedSummarizer()
    policy = ContextPolicy(max_tokens=3, estimator=FlatEstimator(), summarizer=summarizer)

    result = policy.apply(conversation(4))

    assert result.stats.had_summary is False


def test_policy_skips_summary_when_nothing_was_dropped():
    summarizer = FixedSummarizer()

    result = ContextPolicy(max_tokens=4000, summarizer=summarizer).apply(conversation(1))

    assert result.stats.had_summary is False
    assert summarizer.calls == []


def test_policy_ignores_an_empty_summary():
    result = ContextPolicy(max_tokens=120, summarizer=FixedSummarizer(text="")).apply(
        conversation(10)
    )

    assert result.stats.had_summary is False


def test_policy_reports_token_savings():
    result = ContextPolicy(max_tokens=60).apply(conversation(10))

    assert result.stats.saved > 0
    assert result.stats.tokens_before > result.stats.tokens_after


def test_policy_stats_serialise_for_tracing():
    stats = ContextPolicy(max_tokens=60).apply(conversation(10)).stats

    data = stats.to_dict()

    assert set(data) == {
        "tokens_before",
        "tokens_after",
        "kept",
        "dropped",
        "had_summary",
        "truncated",
    }


def test_policy_trim_returns_messages_only():
    trimmed = ContextPolicy(max_tokens=60).trim(conversation(10))

    assert isinstance(trimmed, tuple)
    assert all(isinstance(m, Message) for m in trimmed)


def test_policy_is_deterministic():
    policy = ContextPolicy(max_tokens=60)

    first = policy.apply(conversation(10)).messages
    second = policy.apply(conversation(10)).messages

    assert first == second


def test_policy_rejects_an_empty_conversation():
    with pytest.raises(ValueError):
        ContextPolicy().apply([])


def test_policy_rejects_a_useless_budget():
    with pytest.raises(ValueError):
        ContextPolicy(max_tokens=0)


def test_policy_rejects_a_useless_summary_length():
    with pytest.raises(ValueError):
        ContextPolicy(summary_max_chars=0)


# ---------------------------------------------------------------- ConversationMemory


@pytest.fixture
def memory() -> ConversationMemory:
    return ConversationMemory()


def test_memory_remembers_and_recalls(memory):
    memory.remember("the project uses Python 3.13", tags=["project"])

    hits = memory.recall("Python version")

    assert len(hits) == 1


def test_memory_deduplicates_by_content(memory):
    memory.remember("same fact")

    assert memory.remember("same fact") is None
    assert len(memory.store) == 1


def test_memory_stores_a_turn_as_one_record(memory):
    records = memory.remember_turn("what is 2+3", "5")

    assert len(records) == 1
    assert "Q: what is 2+3" in records[0].text
    assert "A: 5" in records[0].text


def test_memory_skips_empty_answers(memory):
    assert memory.remember_turn("question", "   ") == ()
    assert len(memory.store) == 0


def test_memory_remembers_message_batches(memory):
    added = memory.remember_messages(
        [user_message("hello"), assistant_message("hi"), assistant_message("")]
    )

    assert added == 2


def test_memory_skips_tool_and_system_messages_by_default(memory):
    added = memory.remember_messages(
        [
            system_message("you are terse"),
            Message(role=Role.TOOL, name="search", content="raw output"),
        ]
    )

    assert added == 0


def test_memory_can_store_tool_output_on_request(memory):
    added = memory.remember_messages(
        [Message(role=Role.TOOL, name="search", content="raw output")],
        include_tool_output=True,
    )

    assert added == 1


def test_memory_recall_with_no_hits_is_empty(memory):
    memory.remember("kiln")

    assert memory.recall_messages("blockchain") == ()


def test_memory_recall_message_lists_hits(memory):
    memory.remember("the kiln runs at 1300C")

    messages = memory.recall_messages("kiln temperature")

    assert len(messages) == 1
    assert messages[0].role is Role.SYSTEM
    assert "1300C" in messages[0].content


def test_memory_recall_truncates_long_snippets(memory):
    memory.remember("kiln " + "z" * 300)

    message = memory.recall_messages("kiln", max_chars=20)[0]

    assert "..." in message.content


def test_memory_prime_prepends_system_and_recall(memory):
    memory.remember("the kiln runs at 1300C")

    messages = memory.prime("kiln temperature?", system="be terse")

    assert [m.role.value for m in messages] == ["system", "system", "user"]
    assert messages[0].content == "be terse"


def test_memory_prime_can_skip_recall(memory):
    memory.remember("the kiln runs at 1300C")

    messages = memory.prime("kiln temperature?", recall=False)

    assert [m.role.value for m in messages] == ["user"]


def test_memory_context_applies_the_policy():
    memory = ConversationMemory(policy=ContextPolicy(max_tokens=60))

    result = memory.context(conversation(10))

    assert result.stats.dropped > 0


def test_memory_repr_is_readable(memory):
    memory.remember("a")

    assert "1" in repr(memory)


# ---------------------------------------------------------------- Agent 集成


def test_agent_trims_requests_but_keeps_full_history():
    provider = ScriptedProvider(
        [tool_call_message("big"), tool_call_message("big", call_id="c2"), "done"]
    )
    agent = ReActAgent(
        provider,
        [big],
        config=AgentConfig(context_policy=ContextPolicy(max_tokens=40)),
    )

    result = agent.run("run the big tool")

    assert result.answer == "done"
    assert len(provider.requests[-1].messages) < len(result.messages)


def test_agent_emits_a_trim_event():
    provider = ScriptedProvider(
        [tool_call_message("big"), tool_call_message("big", call_id="c2"), "done"]
    )
    agent = ReActAgent(
        provider,
        [big],
        config=AgentConfig(context_policy=ContextPolicy(max_tokens=40)),
    )

    result = agent.run("run the big tool")

    assert any(e.type == EventType.CONTEXT_TRIMMED for e in result.trace)


def test_agent_without_a_policy_sends_everything():
    provider = ScriptedProvider(
        [tool_call_message("big"), tool_call_message("big", call_id="c2"), "done"]
    )
    agent = ReActAgent(provider, [big])

    result = agent.run("run the big tool")

    # 最后一次请求发出时，最终答案还没产生，所以差一条
    assert len(provider.requests[-1].messages) == len(result.messages) - 1


def test_agent_injects_recalled_memory():
    memory = ConversationMemory()
    memory.remember("the project pins Python 3.13")
    provider = ScriptedProvider(["ok"])

    ReActAgent(provider).run("which python version does the project use?", memory=memory)

    request = provider.last_request
    assert request is not None
    assert any("Python 3.13" in m.content for m in request.messages)


def test_agent_records_a_recall_event():
    memory = ConversationMemory()
    memory.remember("the project pins Python 3.13")

    result = ReActAgent(ScriptedProvider(["ok"])).run(
        "which python version?", memory=memory
    )

    assert any(e.type == EventType.MEMORY_RECALL for e in result.trace)


def test_agent_stores_the_turn_back_into_memory():
    memory = ConversationMemory()

    ReActAgent(ScriptedProvider(["5"])).run("what is 2+3", memory=memory)

    hits = memory.recall("2+3")
    assert hits and "A: 5" in hits[0].text


def test_agent_without_memory_stores_nothing():
    memory = ConversationMemory()

    ReActAgent(ScriptedProvider(["5"])).run("what is 2+3")

    assert len(memory.store) == 0


def test_agent_does_not_store_failed_runs():
    memory = ConversationMemory()
    # 脚本耗尽 => 空回复 => EMPTY，没有答案可存
    agent = ReActAgent(ScriptedProvider([], repeat_last=False), [add])

    agent.run("nothing will come back", memory=memory)

    assert len(memory.store) == 0
