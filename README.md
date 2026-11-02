# nexus-agents

一个**小到能一口气读完、但足够工程化**的 LLM Agent 运行时。

用 Python 从零实现 ReAct 循环、工具调用、Schema 生成、Provider 抽象和链路追踪 ——
不依赖任何 Agent 框架，`pydantic` + `httpx` 就是全部运行时依赖。

> 状态：v0.1.0，核心循环已可用并通过 121 个单元测试（94% 覆盖）。多智能体编排、RAG、FastAPI 服务层仍在开发中，见 [路线图](#路线图)。

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

---

## 测试与质量门禁

```bash
pytest                                            # 121 个测试，全部离线
pytest --cov=nexus_agents --cov-report=term-missing
ruff check src tests
mypy src/nexus_agents
```

当前状态：

- **121 passed**，平均耗时 < 1 秒（无网络、无 API Key、无 sleep）
- 覆盖率 **94%**
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
│   └── providers/
│       ├── base.py        GenerateRequest / LLMResponse / LLMProvider 协议
│       ├── scripted.py    确定性测试替身
│       └── openai_compat.py  OpenAI 兼容 HTTP 后端
├── tests/                 121 个单元测试
├── examples/quickstart.py 离线可跑的端到端示例
└── docs/architecture.md   设计说明
```

---

## 路线图

- [x] v0.1.0 消息模型、工具层、ReAct 循环、Provider 抽象、追踪
- [ ] v0.2.0 状态图编排（`Graph`）：条件边、循环、checkpoint，支撑多智能体
- [ ] v0.3.0 记忆层：窗口裁剪 + 摘要压缩
- [ ] v0.4.0 FastAPI 服务层：SSE 流式输出、会话持久化
- [ ] v0.5.0 RAG 模块：切分、向量检索、重排、评测集
- [ ] v0.6.0 MCP 客户端：把外部 MCP 工具接进 `ToolRegistry`

---

## 安全

- API Key 只从环境变量或显式参数读取，**不写进异常信息、日志或 trace**
- 密钥相关的错误只提示缺哪个环境变量，不回显任何片段
- 工具异常会被捕获成结构化结果，原始堆栈不会泄漏给模型

---

## 许可证

MIT © Jiang Longhuan
