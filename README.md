# nexus-agents

[![CI](https://github.com/jpk06/nexus-agents/actions/workflows/ci.yml/badge.svg)](https://github.com/jpk06/nexus-agents/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

一个**小到能一口气读完、但足够工程化**的 LLM Agent 运行时。

用 Python 从零实现 ReAct 循环、工具调用、Schema 生成、Provider 抽象和链路追踪 ——
不依赖任何 Agent 框架，`pydantic` + `httpx` 就是全部运行时依赖。

> 状态：v0.4.0，ReAct 循环 + 状态图编排 + 记忆层 + 异步运行时 + 事件流式已可用，通过 365 个单元测试（97% 覆盖，`agent.py` / `graph.py` / `memory.py` / `checkpoint.py` 100%）。FastAPI 服务层、RAG、MCP 仍在开发中，见 [路线图](#路线图)。

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
| 事件流式 | `agent.astream()` 边跑边产出事件（工具调用、结果、耗时），最后一项是完整结果；SSE 有真东西可推，不是攒完再切片 |
| HTTP 服务 | `create_app(agent)` 出一个会话级 API：SSE 流式 + 心跳、同会话串行（忙则 409）、TTL/LRU 会话存储、可选 API Key |

---

## 安装

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 想跑 HTTP 服务层再加：
pip install -e ".[api]"
```

需要 Python 3.11+。核心运行时只依赖 `pydantic` + `httpx`；FastAPI 是**可选**依赖，
不会因为「只想用 Agent 循环」就被迫装一整套 web 框架。

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
| — | `agent.astream()` | 事件级流式；`arun` 就是把它抽干 |
| `graph.invoke()` | `graph.ainvoke()` | 节点可以是 `async def` |
| `graph.resume()` | `graph.aresume()` | 断点续跑同样支持异步节点 |
| `registry.invoke()` | `registry.ainvoke()` | 异步工具在这里被 await |
| `provider.generate()` | `provider.agenerate()` | 可选；不实现也能跑 |

### 流式：事件级，不是 token 级

```python
async for item in agent.astream("还有多少茶杯?"):
    if isinstance(item, AgentResult):
        print("done:", item.answer)
        continue
    print(item.type, item.payload)   # agent.start / llm.call / tool.call / tool.result ...
```

**它不假装流 token。** 把已完成的答案切片吐出来，演示很好看，生产里用户照样
要等完整回答才见到第一个字。真 token 流需要一个能产出 delta 的 provider 契约
（增量拼装 `tool_calls` 是最麻烦的一段），那是独立一块工作；落地时它接进同一个
循环，只多一个事件类型。

三条契约，都是会踩坑的地方：

- **事件顺序与 `result.trace` 完全一致** —— 边看边流和跑完再看不会给出两套事实
- **最后一项一定是 `AgentResult`**，消费方靠 `isinstance` 收尾
- **中途 `break` 等于取消**：生成器关闭，这一轮不写记忆（半途而废的问答不该变成记忆）

背压是真的：事件直接产出，不经过无界队列。消费得慢，agent 就跟着慢 —— 否则
一个卡住的 SSE 客户端能把内存吃满。

```bash
python examples/stream_demo.py   # 事件时间线 / 慢模型 / 中途放弃 / SSE 形状
```

---

## HTTP 服务层

```python
# app.py —— 装配和定义分开：路由不认识你的 provider
import os

from nexus_agents import ReActAgent
from nexus_agents.providers import OpenAICompatibleProvider
from nexus_agents.service import ServiceSettings, create_app

agent = ReActAgent(OpenAICompatibleProvider(model="gpt-4o-mini"), tools=[])
app = create_app(agent, settings=ServiceSettings(api_key=os.environ["SERVICE_KEY"]))
```

```bash
pip install -e ".[api]"
uvicorn app:app --reload
curl -s localhost:8000/healthz
curl -sN -X POST localhost:8000/sessions/$SID/messages/stream \
  -H 'Content-Type: application/json' -d '{"text":"还有多少茶杯?"}'
```

| 路由 | 作用 |
| --- | --- |
| `GET /healthz` | 存活探针，不碰模型也不碰存储（探针不能自己变成故障点） |
| `POST /sessions` | 建会话，可带会话级 `system` |
| `GET /sessions/{id}` | 这一会话说过什么、聊了几轮（审计用） |
| `POST /sessions/{id}/messages` | 跑一轮，返回完整 `AgentResult`（含 messages 与 trace） |
| `POST /sessions/{id}/messages/stream` | 同一轮，SSE 逐事件推送 |
| `DELETE /sessions/{id}` | 结束会话 |

四个不显然但要紧的决定：

- **SSE 走 POST。** 浏览器原生 `EventSource` 只能 GET、不能带请求体，也不能加鉴权头；
  把问题塞进查询串既难看又有长度上限。所以用 POST + `text/event-stream`，客户端用
  `fetch` 读流。代价是浏览器端要自己解析帧（十几行），收益是请求体和鉴权头都能用。
- **错误分两层暴露。** 流一开始，状态码就定了 —— 上游模型中途挂掉，你**没法**把 200
  改成 502，只能补一个 `error` 帧。所以 JSON 路径映射状态码（上游故障 502），SSE 路径
  把错误发进流里；不区分就会出现「客户端收到 200 + 半截答案还以为成功了」。
- **心跳不取消正在等的那次拉取。** `asyncio.wait_for(anext(...))` 超时会取消 `anext`，
  等于往 agent 的生成器里扔 `CancelledError` —— 心跳一跳，整轮推理就死了。这里用
  `asyncio.wait` 只等「结果到没到」，超时发一行 `: keep-alive`，同一个 task 继续挂着；
  任何时刻只有一个未完成的拉取，背压依然真实。
- **同一会话并发 = 409，不排队。** 两轮交错会把记忆写乱；而无上限的排队就是内存放大器
  （一个人狂点发送，服务替他攒下几百个待跑的模型调用）。409 让客户端立刻知道「上一个
  还没回」，这是更诚实的背压。

会话状态就在 `Session` 里：`memory`（跨会话召回）、`messages`（逐字历史）、`lock`
（串行化）。**历史和记忆是两件事** —— 历史负责「接得上」（「那它呢」也有主语），
记忆负责「想得起」（跨会话命中）。只有记忆的聊天服务，第二句就失忆。

```bash
python examples/service_demo.py   # 进程内跑完整条链路，不需要端口与网络
```

---

## 性能基准

`benchmarks/benchmark.py` 全程离线：用可控延迟的 provider 和阻塞工具替代真实
LLM，测的是**运行时自身**的开销（调度、线程池分派、会话锁、SSE 帧编码）。
基准自带护栏断言 —— 「工具参数写错导致校验阶段 0ms 失败」这类静默 bug 会让
数字变小变漂亮，护栏直接把它拦下来。

本机实测（Python 3.13.14 / Windows 11 / 16 核，2026-11-05，每轮 = 2 次模型调用
+ 1 次工具调用）：

| 指标 | 数值 |
| --- | --- |
| 每轮框架开销（provider 立即返回） | P50 **0.10 ms** / P99 0.31 ms |
| 10 轮并发 vs 串行（每轮约 185ms，含阻塞工具） | **9.96x**，并发墙钟 ≈ 单轮 |
| `astream` 首帧延迟（provider 延迟 80ms） | **< 0.1 ms**（不用等整轮就能画进度条） |
| HTTP 一轮往返（进程内 ASGI，含会话/历史/锁） | P50 **1.3 ms** / P99 3.3 ms |
| HTTP 10 会话并发 vs 串行 | **9.42x**（会话锁只锁同一会话，不锁别人） |

复现：`python benchmarks/benchmark.py`（约 8 秒，无 API Key、无网络、无端口）。
数字测的是框架不是模型 —— 真实部署的延迟大头永远是 LLM 本身。

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
pytest                                            # 365 个测试，全部离线
pytest --cov=nexus_agents --cov-report=term-missing
ruff check src tests examples
mypy src/nexus_agents
```

当前状态：

- **365 passed**，平均耗时 ~2 秒（无网络、无 API Key、无真实等待）
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
│   ├── service/           可选依赖层：HTTP 服务（FastAPI）
│   │   ├── app.py         create_app 工厂 / 路由 / SSE 生成器
│   │   ├── sessions.py    Session + 存储协议 + TTL/LRU 内存实现
│   │   └── sse.py         SSE 帧编码（纯函数，好测）
│   └── providers/
│       ├── base.py        GenerateRequest / LLMResponse / LLMProvider 协议
│       ├── scripted.py    确定性测试替身
│       └── openai_compat.py  OpenAI 兼容 HTTP 后端
├── tests/                 413 个单元测试
├── examples/
│   ├── quickstart.py      单 Agent 端到端示例（离线）
│   ├── graph_demo.py      循环 / 条件边 / checkpoint / Mermaid
│   ├── multiagent.py      supervisor + worker 多智能体
│   ├── memory_demo.py     上下文裁剪 / 摘要压缩 / 中文长期记忆
│   ├── async_demo.py      同步/异步一致性 / 并发 / 不阻塞事件循环
│   ├── stream_demo.py     事件时间线 / 慢模型 / 中途放弃 / SSE 形状
│   └── service_demo.py    HTTP 层：会话 / SSE / 409 / 会话视图
├── benchmarks/
│   └── benchmark.py       离线性能基准（开销 / 并发 / 首帧 / HTTP 层）
└── docs/architecture.md   设计说明
```

---

## 路线图

- [x] v0.1.0 消息模型、工具层、ReAct 循环、Provider 抽象、追踪
- [x] v0.2.0 状态图编排（`StateGraph`）：条件边、循环、checkpoint，支撑多智能体
- [x] v0.3.0 记忆层：token 预算裁剪（工具调用原子性）、摘要压缩、BM25-lite 长期记忆
- [x] v0.4.0 异步运行时：`arun` / `ainvoke` / `aresume`，同步与异步共用同一控制流；含事件级流式 `astream`
- [x] v0.5.0 FastAPI 服务层：SSE（建在 `astream` 之上，含心跳与取消）、会话级历史与隔离、TTL/LRU 存储
- [ ] v0.6.0 RAG 模块：切分、向量检索、重排、评测集
- [ ] v0.7.0 MCP 客户端：把外部 MCP 工具接进 `ToolRegistry`

---

## 安全

- API Key 只从环境变量或显式参数读取，**不写进异常信息、日志或 trace**
- 密钥相关的错误只提示缺哪个环境变量，不回显任何片段
- 工具异常会被捕获成结构化结果，原始堆栈不会泄漏给模型
- 服务层：未知字段一律 422（不静默忽略），输入有长度上限，错误帧只回显自家异常，
  陌生异常的 message 绝不外泄；`X-API-Key` 用 `compare_digest` 比较

---

## 许可证

MIT © Jiang Longhuan
