"""Memory demo: trim a long conversation, compress the overflow, recall old facts.

Run offline::

    python examples/memory_demo.py

三个场景，全部用脚本化 provider 驱动，不需要 API Key：

1. 上下文裁剪 —— 一场长对话塞不进预算时，丢掉最老的部分，但**绝不拆开**
   assistant(tool_calls) 和它的 tool 结果（拆开就等于 400）。
2. 摘要压缩 —— 被丢掉的那段压成一条 system 消息，模型仍然知道前面发生过什么。
3. 长期记忆 —— 跨会话的事实用关键词检索捞回来，自动注入下一次请求。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus_agents import (
    SUMMARY_PREFIX,
    AgentConfig,
    ContextPolicy,
    ConversationMemory,
    ExtractiveSummarizer,
    HeuristicTokenEstimator,
    InMemoryMemoryStore,
    ReActAgent,
    tool,
)
from nexus_agents.providers import ScriptedProvider
from nexus_agents.types import (
    Message,
    Role,
    ToolCall,
    assistant_message,
    system_message,
    user_message,
)

ESTIMATOR = HeuristicTokenEstimator()


@tool
def kiln_log(batch: str) -> str:
    """Read the firing log for one batch.

    Args:
        batch: batch identifier
    """
    return (
        f"batch {batch}: peak 1300C, hold 40min, reduction at 1100C, "
        "12 pieces, 2 cracked on the rim, glaze slightly underfired on the foot ring"
    )


# ---------------------------------------------------------------- 1. 长对话裁剪


def long_conversation(turns: int = 8):
    messages = [system_message("You are a kiln technician. Be precise about numbers.")]
    for i in range(turns):
        messages.append(user_message(f"Batch B{i}: what was the peak temperature and hold time?"))
        messages.append(
            assistant_message(f"Batch B{i} peaked at {1280 + i}C with a 40 minute hold.")
        )
    messages.append(user_message("Summarise what changed across the batches."))
    return messages


def show_trimming() -> None:
    messages = long_conversation()
    policy = ContextPolicy(max_tokens=180)
    result = policy.apply(messages)

    print("=== 1. context trimming ===")
    print("budget          : 180 tokens")
    print(f"before          : {result.stats.tokens_before} tokens / {len(messages)} messages")
    print(f"after           : {result.stats.tokens_after} tokens / {len(result.messages)} messages")
    print(f"dropped         : {result.stats.dropped} messages")
    print(f"over budget     : {result.stats.truncated}")
    print()
    for message in result.messages:
        label = message.name or message.role.value
        print(f"  keep [{label:9}] {message.content[:64]}")
    print()


# ---------------------------------------------------------------- 2. 摘要压缩


def show_summary() -> None:
    messages = long_conversation()
    policy = ContextPolicy(
        max_tokens=180,
        summarizer=ExtractiveSummarizer(max_lines=4),
        summary_max_chars=180,
    )
    result = policy.apply(messages)

    print("=== 2. summary compression ===")
    print(f"had_summary     : {result.stats.had_summary}")
    print(f"dropped         : {result.stats.dropped} messages -> 1 summary message")
    for message in result.messages:
        if message.content.startswith(SUMMARY_PREFIX):
            print("  summary block:")
            for line in message.content.splitlines():
                print(f"    {line}")
    print()


# ---------------------------------------------------------------- 3. 工具配对


def tool_msg(call_id: str, name: str, content: str) -> Message:
    return Message(role=Role.TOOL, tool_call_id=call_id, name=name, content=content)


def show_tool_pairing() -> None:
    messages = [
        system_message("You are a kiln technician."),
        user_message("Check batch B7 and tell me the peak temperature."),
        assistant_message(tool_calls=(ToolCall(id="call_1", name="kiln_log", arguments={}),)),
        tool_msg("call_1", "kiln_log", "peak 1300C, hold 40min"),
        user_message("Now do the same for B8, and compare the two."),
    ]
    trimmed = ContextPolicy(max_tokens=60).apply(messages).messages

    print("=== 3. tool pairing is atomic ===")
    print(f"input messages  : {[m.role.value for m in messages]}")
    print(f"kept messages   : {[m.role.value for m in trimmed]}")
    print("rule            : assistant(tool_calls) + its tool results are dropped together")
    print()


# ---------------------------------------------------------------- 4. 长期记忆


def show_long_term_memory() -> None:
    store = InMemoryMemoryStore(recency_weight=0.2)
    memory = ConversationMemory(
        store=store,
        policy=ContextPolicy(max_tokens=400, summarizer=ExtractiveSummarizer(max_lines=3)),
    )

    # 前几次会话沉淀下来的事实（真实系统里来自数据库/文件）
    memory.remember("客户要求所有茶壶的釉色偏青，参考宋代汝窑。", tags=["preference"])
    memory.remember("景德镇到上海的物流时效是 2 天，成本 18 元/件。", tags=["logistics"])
    memory.remember("上个月退货率 3.2%，主要集中在壶嘴开裂。", tags=["quality"])

    print("=== 4. long-term recall ===")
    print(f"stored records  : {len(store)}")
    for query in ("汝窑釉色要求", "上海 物流 成本", "退货率"):
        hits = memory.recall(query, k=1)
        print(f"  query {query!r:16} -> {hits[0].text if hits else '(no hit)'}")

    # 召回结果会被自动拼进下一次请求
    provider = ScriptedProvider(
        [lambda request: "按记录：汝窑偏青，物流 2 天 18 元/件，退货率 3.2%。"]
    )
    agent = ReActAgent(provider, [kiln_log], config=AgentConfig(max_steps=3))
    result = agent.run("客户偏好的釉色是什么？顺便说一下物流和退货情况。", memory=memory)

    request = provider.last_request
    injected = [m for m in request.messages if m.role is Role.SYSTEM and "汝窑" in m.content]
    print()
    print(f"recall injected : {len(injected)} system message(s)")
    print(f"answer          : {result.answer}")
    print(f"memory after run: {len(store)} records (this turn was stored too)")
    print()


def main() -> None:
    show_trimming()
    show_summary()
    show_tool_pairing()
    show_long_term_memory()

    print("=== token estimate sanity check ===")
    for text in ("hello world", "景德镇陶瓷", "mixed 中英 text"):
        print(f"  {text!r:20} -> {ESTIMATOR.count(text)} tokens (estimate)")


if __name__ == "__main__":
    main()
