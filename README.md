# nexus-agents

[![CI](https://github.com/jpk06/nexus-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/jpk06/nexus-agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个**小到能一口气读完、但足够工程化**的 LLM Agent 运行时。

用 Python 从零实现 ReAct 循环、工具调用、Schema 生成、Provider 抽象和链路追踪 ——
不依赖任何 Agent 框架，`pydantic` + `httpx` 就是全部运行时依赖。

> 状态：v0.4.0，ReAct 循环 + 状态图编排 + 记忆层 + 异步运行时已可用，通过 353 个单元测试（97% 覆盖，`agent.py` / `graph.py` / `memory.py` / `checkpoint.py` 100%）。FastAPI 服务层、RAG、MCP 仍在开发中，见 [路线图](#路线图)。

---

## 为什么要造这个轮子

用框架写 Agent 很快，但你会不知道：

- 模型看到的工具描述到底长什么样
- 工具报错时循环是怎么恢复的
- 一次运行到底调了几次模型、花了多少 token
- 跑满步数时系统是「假装成功」还是如实上报

这些恰恰是生产事故高发的地方。所以这个项目的取舍是：**把每一层都做得足够薄，薄到能读完、能改、能在测试里精确断言。**

它不是 LangGraph / AutoGen 的替代品，而是它们的「解剖图谱」。

---

## 特性

| 能力 | 说明 |
| --- | --- |
| 工具定义 | `@tool` 从类型注解生成 JSON Schema，从 docstring 生成描述，**代码与描述不可能脱节** |
| 参数校验 | 模型返回 `"7"` 而函数要 `int`？自动强转；多传字段、少传字段、非法枚举值都会被拦下 |
| 错误隔离 | 工具抛异常不会炸掉 Agent —— 异常转成 `ToolResult(is_error=True)` 回喂模型，让它自我纠错 |
| Provider 抽象 | Protocol 而非基类：`OpenAICompatibleProvider` 通吃 OpenAI / vLLM / Ollama / DeepSeek，`ScriptedProvider` 让测试零成本零网络 |
| 可观测 | 每次运行产出完整事件流（含 LLM 调用耗时），可断言、可序列化 |
| 成本可见 | token 用量跨步累加，直接挂在结果上 |
| 诚实终止 | 跑满 `max_steps` 一定报 `max_steps`，绝不伪装成成功 |
| 状态图编排 | `StateGraph`：条件边、循环、回边，节点是纯函数，边是数据，图可校验可导出 Mermaid |
| 断点续跑 | 每一步都写 checkpoint，`resume(thread_id)` 从中断处继续 —— 人工审批、长流程都不会丢状态 |
| 多智能体 | supervisor/worker 只是图的一种写法：worker 仍是普通 `ReActAgent`，完全不知道自己被编排 |
| 上下文裁剪 | 按 token 预算裁剪历史；**工具调用与它的结果永远同进同出**，不会拼出非法请求 |
| 摘要压缩 | 被裁掉的老对话压成一条 system 摘要，模型仍知道前面发生过什么；可离线（抽取式）也可用模型 |
| 长期记忆 | 跨会话事实按 BM25-lite 检索（支持中文 bigram），自动注入下一次请求，零向量库依赖 |
| 异步运行时 | `arun` / `ainvoke` / `aresume` 全套异步入口；同步与异步共用**同一份控制流**，不会各修各的 |
| 不阻塞 | 同步 provider 与同步工具自动进线程池；一个 `time.sleep` 的工具不会让整个服务失去响应 |

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

需要 Python 3.11+。

---

## 快速上手

### 1. 定义工具

```python
from nexus_agents import tool

@tool
def get_price(product: str) -> str:
    """Look up the price of a product.

    Args:
        product: exact product name
    """
    return {"celadon vase": "¥1,280"}.get(product, "unknown product")
```

这一行 `@tool` 生成的 Schema：

```json
{
  "type": "object",
  "properties": {
    "product": { "type": "string", "description": "exact product name" }
  },
  "required": ["product"],
  "additionalProperties": false
}
```

支持 `str` / `int` / `float` / `bool`、`list[T]`、`dict[K, V]`、`tuple[T, ...]`、`Enum`、`Literal`、`Optional[T]`、`X | Y`、pydantic 模型，以及 Google 和 numpy 两种 docstring 风格。

### 2. 跑一次 Agent

```python
from nexus_agents import ReActAgent, ToolRegistry
from nexus_agents.providers import OpenAICompatibleProvider
import os

provider = OpenAICompatibleProvider(
    model="gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
)

agent = ReActAgent(provider, ToolRegistry([get_price]))
result = agent.run("How much is the celadon vase?")

print(result.answer)          # 最终答案
print(result.stopped_reason)  # Stopped.FINAL / MAX_STEPS / TOOL_ERROR / EMPTY
print(result.steps)           # 实际用了几轮
print(result.usage.total_tokens)
```

### 3. 不用 API Key 也能试

```bash
python examples/quickstart.py
```

示例用 `ScriptedProvider` 模拟一个会「先搜索、再查价、最后回答」的模型，**完全离线**，输出完整对话与工具链路。

---

## 状态图编排

单个 Agent 适合做一件事，真实任务通常是「检索 → 判断要不要再检索 → 写作 → 审校 → 不过就重写」。
把这些写进 `if/elif` 能跑，但会迅速变成没人敢改的意大利面。`StateGraph` 把流程显式化：

```python
from nexus_agents import END, InMemoryCheckpointer, StateGraph, append

graph = StateGraph({"evidence": append})          # evidence 累积，其它字段覆盖

graph.add_node("search", search)                  # 节点是纯函数：(state) -> 局部更新
graph.add_node("write", write)
graph.add_conditional_edges(                      # 路由函数决定下一步去哪
    "search",
    lambda state: "write" if len(state["evidence"]) >= 3 else "search",
    {"search": "search", "write": "write"},
)
graph.set_entry_point("search")

compiled = graph.compile(checkpointer=InMemoryCheckpointer(), max_steps=8)
result = compiled.invoke({"evidence": []}, thread_id="run-1")

result.path          # ('search', 'search', 'write')
result.status        # GraphStatus.COMPLETED / MAX_STEPS / ERROR
result.state         # 最终状态（只读视图）
```

四个关键约定：

| 约定 | 为什么 |
| --- | --- |
| 节点返回**局部更新**，不是整个新状态 | 节点之间只通过约定字段通信，互不认识 |
| 需要累积的字段显式注册 reducer | 否则两个节点写同一个字段会互相覆盖（默认最后写入生效） |
| 节点拿到的是**只读视图** | 直接改状态会当场 `TypeError`，而不是悄悄污染下游节点 |
| `compile()` 时校验 | 入口存在、边指向真实节点、没有永远走不到的节点 —— 跑不起来的图根本编译不过 |

### 断点续跑

```python
partial = graph.compile(checkpointer=store, max_steps=2).invoke(state, thread_id="t1")
# partial.status is GraphStatus.MAX_STEPS，但前两步的状态已经存好

continued = graph.compile(checkpointer=store, max_steps=20).resume("t1")
```

`Checkpointer` 是个 Protocol：内存实现用于测试和单进程，换成 Redis / Postgres 只要实现
`save` / `load` / `history` / `threads` / `clear`，图的代码一行都不用改。

### 多智能体

supervisor/worker 不需要新的抽象 —— worker 就是普通 `ReActAgent`，supervisor 只是一个
「看状态、决定下一步派谁」的节点：

```python
graph.add_node("supervisor", lambda state: {})   # 只负责路由
graph.add_conditional_edges("supervisor", route, {"research": "research", "write": "write", ...})
for worker in ("research", "write", "review"):
    graph.add_edge(worker, "supervisor")          # 干完回到 supervisor
```

```bash
python examples/graph_demo.py   # 循环 + 条件边 + checkpoint + Mermaid 导出
python examples/multiagent.py   # supervisor + 三个 worker agent，审校不通过自动重写
```

---

## 记忆层

Agent 跑久了只有两个结局：上下文撑爆，或者历史被一刀切干净。v0.3.0 把这两件事分开处理：

```python
from nexus_agents import ContextPolicy, ConversationMemory, ExtractiveSummarizer

memory = ConversationMemory(
    policy=ContextPolicy(
        max_tokens=4000,                       # 发给模型的硬预算
        summarizer=ExtractiveSummarizer(),     # 被裁掉的部分压成一条摘要（离线可用）
    )
)

result = agent.run("客户偏好的釉色是什么？", memory=memory)
#   ├─ 先按关键词召回相关长期记忆，作为 system 消息注入
#   ├─ 每步调模型前按预算裁剪上下文
#   └─ 跑完把这一轮问答写回记忆
```

三个容易被忽略、但真会出事的地方：

| 问题 | 这里的处理 |
| --- | --- |
| `role=tool` 消息必须跟在带 `tool_calls` 的 assistant 后面，否则 API 直接 400 | 裁剪以「assistant + 它的全部 tool 结果」为**原子单元**，要么全留、要么全丢 |
| 贪心填充预算会把空间吃光，摘要永远差一点塞不进去 | 启用摘要时先预留 1/4 预算给它 |
| 裁完窗口从一句「没头没尾的 assistant 回复」开始 | 带 `tool_calls` 的保留（后面挂着结果），裸回复连同它回答的问题一起丢 |

中文检索不引入 jieba：连续的 CJK 串切成字符 bigram（`景德镇陶瓷` → 景德 / 德镇 / 镇陶 / 陶瓷），
配合 idf × 词频饱和 × 长度归一打分，几百到几万条记忆足够用，结果还是确定的 —— 测试可以断言排名。

```bash
python examples/memory_demo.py   # 长对话裁剪 + 摘要 + 中文召回，全离线
```

---

## 异步

一个 Agent 服务同时接 50 个用户时，最致命的不是慢，是**一个慢请求卡住所有人**。
所以异步不是「加个 async 关键字」，而是三件事都得成立：

```python
import asyncio
from nexus_agents import ReActAgent, tool

@tool
async def fetch_price(sku: str) -> str:
    """Query a supplier API."""
    async with client.get(f"/price/{sku}") as resp:
        return await resp.text()

agent = ReActAgent(provider, [fetch_price])

# 一次请求
result = await agent.arun("多少钱？")

# 并发 50 个会话，各自独立推进
results = await asyncio.gather(*(agent.arun(q) for q in questions))
```

三件事分别是：

**1. 只有一个循环。** `arun` 是真正的实现，`run` 只是 `asyncio.run(arun(...))`。
写成两份（同步一份、异步一份）的框架，一周之内就会分叉 —— 有人在一边修了
停止条件，忘了另一边，于是两条路径对「Agent 该不该停」给出不同答案。代价是
`run()` 不能在有事件循环的线程里调用，这种情况它会直接报错并指向 `arun`。

**2. 同步实现也要挪到线程池。** provider 只实现 `generate` 完全可以，`arun`
会把它丢进线程池；同步工具同理。否则一个 `time.sleep(0.05)` 的工具就能让
整个事件循环停摆 —— 这类 bug 在单元测试里看不见，因为它「功能正常，只是慢」。

**3. 测试要能证明「真的没阻塞」。** 光断言「await 能跑通」毫无意义，一个在
async 函数里写 `time.sleep` 的实现照样能过。所以测试用**时间线断言**：慢 IO
在跑的时候，另一个协程必须先完成。

```bash
python examples/async_demo.py   # 结果一致性 / 并发加速 / 慢工具不卡循环
```

| 同步入口 | 异步入口 | 说明 |
| --- | --- | --- |
| `agent.run()` | `agent.arun()` | 同一个循环，同一份结果 |
| `graph.invoke()` | `graph.ainvoke()` | 节点可以是 `async def` |
| `graph.resume()` | `graph.aresume()` | 断点续跑同样支持异步节点 |
| `registry.invoke()` | `registry.ainvoke()` | 异步工具在这里被 await |
| `provider.generate()` | `provider.agenerate()` | 可选；不实现也能跑 |

---

## 核心设计

### 控制流

```
用户输入
   │
   └─► while step < max_steps:
         ├─ provider.generate(对话 + 工具定义)
         ├─ 模型没要调工具 ─────────────► 返回最终答案 (FINAL)
         ├─ 模型要调工具 ─► 逐个执行 ─► 结果作为 role=tool 消息追加回对话
         └─ 工具连续失败 ──────────────► 如实终止 (TOOL_ERROR)
       跑满步数 ────────────────────────► 如实终止 (MAX_STEPS)
```

工具**串行**执行是刻意的选择：并行会让 trace 不确定、错误处理复杂化。并行策略以后可以在同一接口后面加。

### 错误是数据，不是异常

```python
@tool
def boom() -> str:
    """Always raises."""
    raise ValueError("kaboom")

result = boom.invoke(call)
result.is_error   # True
result.error      # "ValueError: kaboom"
```

这条消息会被送回模型。模型看到「我调错了」之后，常见的行为是换参数重试或换一个工具 —— 不需要任何额外代码。

### 每一层都能单独替换

| 层 | 位置 | 换掉的代价 |
| --- | --- | --- |
| 消息模型 | `types.py` | 所有层都要读，改动代价最大 |
| Schema 生成 | `schema.py` | 纯函数，随便改 |
| 工具执行 | `tools.py` | 实现 `invoke()` 即可 |
| LLM 后端 | `providers/` | 实现 `generate(request) -> response` 一个方法 |
| 控制循环 | `agent.py` | 上面四层不用动 |
| 流程编排 | `graph.py` | 节点是纯函数，换编排方式不用动 `agent.py` |
| 状态存储 | `checkpoint.py` | 实现 Checkpointer 协议即可换存储后端 |
| 记忆与上下文 | `memory.py` | 换 token 估算器 / 摘要器 / 存储后端都只是实现一个协议 |

---

## 测试与质量门禁

```bash
pytest                                            # 353 个测试，全部离线
pytest --cov=nexus_agents --cov-report=term-missing
ruff check src tests examples
mypy src/nexus_agents
```

当前状态：

- **353 passed**，平均耗时 ~2 秒（无网络、无 API Key、无真实等待）
- 覆盖率 **97%**（`agent.py`、`graph.py`、`memory.py`、`checkpoint.py`、`types.py`、`tracing.py` 均 100%）
- `ruff` 与 `mypy` 零告警

测试替身是 `ScriptedProvider`：可以回放固定响应，也可以传一个函数按请求动态决定返回什么 —— 这样连「模型看到工具结果后怎么反应」都能精确断言。

HTTP 层用 `httpx.MockTransport` 测试，覆盖正常解析、工具调用解析、401/500、重试退避、畸形响应。

---

## 项目结构

```
nexus-agents/
├── src/nexus_agents/
│   ├── __init__.py        公开 API
│   ├── types.py           Message / ToolCall / ToolResult / Usage
│   ├── errors.py          异常体系
│   ├── schema.py          类型注解 -> JSON Schema，docstring 解析
│   ├── tools.py           @tool / ToolSpec / ToolRegistry
│   ├── tracing.py         事件流追踪
│   ├── agent.py           ReAct 循环
│   ├── memory.py          token 估算 / 检索 / 上下文裁剪 / 摘要
│   ├── checkpoint.py      Checkpoint / Checkpointer 协议 / 内存实现
│   ├── graph.py           StateGraph：条件边、循环、reducer、编译与执行
│   └── providers/
│       ├── base.py        GenerateRequest / LLMResponse / LLMProvider 协议
│       ├── scripted.py    确定性测试替身
│       └── openai_compat.py  OpenAI 兼容 HTTP 后端
├── tests/                 353 个单元测试
├── examples/
│   ├── quickstart.py      单 Agent 端到端示例（离线）
│   ├── graph_demo.py      循环 / 条件边 / checkpoint / Mermaid
│   ├── multiagent.py      supervisor + worker 多智能体
│   ├── memory_demo.py     上下文裁剪 / 摘要压缩 / 中文长期记忆
│   └── async_demo.py      同步/异步一致性 / 并发 / 不阻塞事件循环
└── docs/architecture.md   设计说明
```

---

## 路线图

- [x] v0.1.0 消息模型、工具层、ReAct 循环、Provider 抽象、追踪
- [x] v0.2.0 状态图编排（`StateGraph`）：条件边、循环、checkpoint，支撑多智能体
- [x] v0.3.0 记忆层：token 预算裁剪（工具调用原子性）、摘要压缩、BM25-lite 长期记忆
- [x] v0.4.0 异步运行时：`arun` / `ainvoke` / `aresume`，同步与异步共用同一控制流
- [ ] v0.5.0 FastAPI 服务层：SSE 流式输出、会话持久化
- [ ] v0.6.0 RAG 模块：切分、向量检索、重排、评测集
- [ ] v0.7.0 MCP 客户端：把外部 MCP 工具接进 `ToolRegistry`

---

## 安全

- API Key 只从环境变量或显式参数读取，**不写进异常信息、日志或 trace**
- 密钥相关的错误只提示缺哪个环境变量，不回显任何片段
- 工具异常会被捕获成结构化结果，原始堆栈不会泄漏给模型

---

## 许可证

MIT © Jiang Longhuan
