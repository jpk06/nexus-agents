"""Quickstart: define tools, run an agent, inspect the trace.

Run it with no API key and no network access::

    python examples/quickstart.py

The scripted provider stands in for a real LLM so the control flow is
visible and deterministic. Swap in ``OpenAICompatibleProvider`` (see the
bottom of this file) to talk to a real model.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus_agents import AgentConfig, ReActAgent, ToolRegistry, tool
from nexus_agents.providers import ScriptedProvider, tool_call_message
from nexus_agents.types import Role

# ---------------------------------------------------------------- 1. 定义工具
#
# 工具就是普通函数。类型注解 -> JSON Schema，docstring -> 给模型看的说明。
# 两者都由 @tool 自动生成，永远不会和代码脱节。


@tool
def search_catalogue(query: str, limit: int = 3) -> list[str]:
    """Search the ceramic catalogue for a product.

    Args:
        query: keyword to look for
        limit: how many results to return
    """
    catalogue = ["celadon vase", "blue-and-white bowl", "porcelain teapot"]
    return [item for item in catalogue if query.lower() in item][:limit]


@tool
def get_price(product: str) -> str:
    """Look up the price of a product.

    Args:
        product: exact product name
    """
    prices = {
        "celadon vase": "¥1,280",
        "blue-and-white bowl": "¥640",
        "porcelain teapot": "¥320",
    }
    return prices.get(product, "unknown product")


# ---------------------------------------------------------------- 2. 编排一次运行


def make_scripted_provider():
    """A fake model that knows it should search, then price, then answer."""

    def policy(request):
        saw_tool_results = any(m.role is Role.TOOL for m in request.messages)
        has_price = any(
            m.role is Role.TOOL and "¥" in m.content for m in request.messages
        )
        if has_price:
            return "The celadon vase costs ¥1,280."
        if saw_tool_results:
            return tool_call_message("get_price", {"product": "celadon vase"})
        return tool_call_message("search_catalogue", {"query": "vase", "limit": 1})

    return ScriptedProvider([policy], usage=None)


def main() -> None:
    registry = ToolRegistry([search_catalogue, get_price])
    agent = ReActAgent(
        make_scripted_provider(),
        registry,
        config=AgentConfig(
            system_prompt="You are a ceramics shop assistant. Answer concisely.",
            max_steps=6,
        ),
        name="shop-assistant",
    )

    result = agent.run("How much is the celadon vase?")

    print("answer        :", result.answer)
    print("stopped_reason:", result.stopped_reason.value)
    print("steps         :", result.steps)
    print("tokens        :", result.usage.total_tokens)
    print()

    print("--- conversation ---")
    for message in result.messages:
        detail = message.content or ", ".join(c.name for c in message.tool_calls)
        print(f"  [{message.role.value:9}] {detail}")

    print()
    print("--- tool trace ---")
    for event in result.trace:
        if event.type in ("tool.call", "tool.result"):
            status = "ERROR" if event.payload.get("is_error") else "ok"
            print(f"  step {event.step}  {event.type:12} {event.payload.get('tool'):18} {status}")


# ---------------------------------------------------------------- 3. 接真实模型
#
# from nexus_agents.providers import OpenAICompatibleProvider
#
# provider = OpenAICompatibleProvider(
#     model="gpt-4o-mini",              # 或 vLLM / Ollama 上的任意模型
#     api_key=os.environ["OPENAI_API_KEY"],
# )
# agent = ReActAgent(provider, registry)
# print(agent.run("How much is the celadon vase?").answer)


if __name__ == "__main__":
    main()
