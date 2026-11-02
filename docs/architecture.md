# 架构说明

## 分层与依赖方向

```
        agent.py            控制循环（唯一有「流程」的地方）
           │
    ┌──────┼──────────┐
    ▼      ▼          ▼
tools.py  providers/  tracing.py
    │          │
    ▼          ▼
schema.py   types.py  ←── 所有层共享，依赖最稳定
```

依赖**单向向下**：上层可以用下层，下层永远不知道上层存在。
好处是替换任意一层都不用碰其它层 —— 换 LLM 后端只动 `providers/`，
改循环策略只动 `agent.py`。

## 各层职责

### `types.py` —— 通用货币

`Message` / `ToolCall` / `ToolResult` / `Usage` 全部是 **frozen pydantic 模型**。

- 不可变：一次运行中可以安全地到处传递，不用担心被某层偷偷改掉
- 可序列化：`to_api_payload()` 一处实现 OpenAI 线格式，所有后端复用
- `ToolCall.arguments` 带一个校验器：模型经常把参数传成 JSON 字符串而不是对象，这里统一兜住

### `schema.py` —— 代码即描述

工具描述从**运行时签名**生成，而不是手写的字典。这消除了最常见的
一类 bug：函数加了参数，但给模型的描述忘了改。

两个设计决定：

1. **无法识别的注解返回 `{}` 而不是抛异常。**
   一个冷门类型不该毁掉整个工具注册。宁可让模型看到「任意值」，
   也不要让服务起不来。
2. **`additionalProperties: false`。**
   默认拒绝模型编造未声明的参数，把幻觉挡在执行之前。

docstring 解析同时支持 Google 和 numpy 风格，并且会识别 `Returns:` /
`Raises:` 等段头 —— 否则「Returns」会被当成一个参数名。

### `tools.py` —— 校验 + 隔离

```
模型参数 ──► pydantic 动态模型 ──► 校验并强转 ──► Python 函数
                  │                                    │
                  └─ 失败：InvalidToolArguments          └─ 抛异常
                          │                                    │
                          └──────────► ToolResult(is_error) ◄──┘
```

关键点：**工具的失败是数据，不是控制流**。
`invoke()` 永远返回 `ToolResult`，绝不向上抛。这样 Agent 循环不会被一个
坏工具打断，而是把错误交给模型去补救。

`ToolRegistry` 是模型能碰到工具的唯一入口，因此「某个 Agent 能用哪些工具」
这件事有单一事实来源，`restricted_to()` 可以按角色裁剪工具集。

### `providers/` —— 唯一依赖外部世界的边界

用 `Protocol`（结构化类型）而不是抽象基类：

```python
class LLMProvider(Protocol):
    def generate(self, request: GenerateRequest) -> LLMResponse: ...
```

任何有这个方法的对象都是合法 provider —— 测试替身、缓存装饰器、
fallback 链都不需要继承任何东西。

- `ScriptedProvider`：回放脚本，**也可以传一个函数**按请求动态返回，
  于是「模型看到工具结果之后的反应」也能被精确断言
- `OpenAICompatibleProvider`：支持注入 `httpx.Client`，测试时用
  `MockTransport`，生产时用真实连接；对 429/5xx 做指数退避重试

### `agent.py` —— 循环

只有四件事：调模型、跑工具、判断结束、记账。

三个容易做错的细节，这里都处理了：

1. **连续失败计数跨轮累计**（不是每轮重置）。
   否则「模型反复用同一个坏工具」永远触发不了上限。
2. **跑满步数时回退到最后一条 assistant 文本**，而不是返回空字符串。
3. **`while ... else` 区分「break 出来」和「条件耗尽」**，
   所以 `MAX_STEPS` 不会和 `FINAL` 混淆。

### `tracing.py` —— 事件流

扁平的事件列表，不引入 OpenTelemetry。
理由：当前目标是「看清一次 run 发生了什么」，一条能断言、能序列化的
列表就够了。将来要接 OTel，也只是把事件转成 span。

## 关键取舍

| 决定 | 选了 | 没选 | 为什么 |
| --- | --- | --- | --- |
| 工具执行 | 串行 | 并行 | 并行让 trace 不确定、错误处理复杂化 |
| 工具报错 | 转成数据回喂 | 直接抛 | 让模型自我纠错，比写重试逻辑便宜 |
| Provider | Protocol | ABC | 测试替身不必继承 |
| 未知注解 | 返回 `{}` | 抛异常 | 一个类型不该拖垮整个注册 |
| 追踪 | 自带事件流 | OpenTelemetry | 现阶段不需要分布式追踪的重量 |
| 消息模型 | frozen | 可变 | 一次运行中到处传，可变是隐患 |

## 已知限制

- 单 Agent，跨 Agent 编排要靠 v0.2.0 的状态图
- 无内置记忆管理，长对话会一直变长（v0.3.0 解决）
- 只支持 function-calling 风格的工具调用，不支持从纯文本里解析 ReAct 格式
- 无异步接口（v0.4.0 配合 FastAPI 一起加）
