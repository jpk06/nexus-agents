# 运维手册（Operations Runbook）

> 面向要把 nexus-agents 服务层跑在真实环境里的人：SSE 断线后会发生什么、
> 指标该配什么告警、日志怎么查、环境变量怎么调。所有行为都以代码和测试
> 为准（本文写的每一条都对应 `service/` 里的实现），没有「计划中」的行为。

---

## 1. SSE 流的生命周期与断线语义

### 一轮请求的帧序列

`POST /sessions/{id}/messages/stream` 返回 `text/event-stream`，帧顺序与
agent 的 trace 完全一致：

```
event: agent.start        → data: {step, timestamp, ...}
event: step.start         ┐
event: llm.call           │ 每个推理步一组，
event: llm.response       │ 数量取决于工具调用轮数
event: tool.call          │
event: tool.result        ┘
event: agent.result       → 最后一帧：answer / stopped_reason / usage / steps
```

两种终止之外的帧：

- **`event: error`**：流开始之后 agent 出错（上游模型 5xx、超时等）。此时
  HTTP 状态码早已随响应头发出（必然是 200），错误只能以帧补发 —— 这是
  SSE 的协议约束，不是缺陷。`POST /messages`（非流式版）没有这个约束，
  所以它能把同一错误映射成 502。
- **`: keep-alive`**：注释帧（客户端会忽略），每 `NEXUS_HEARTBEAT` 秒（默认
  15）一条。作用是重置 nginx / 负载均衡的空闲计时，防止「模型思考 30 秒」
  被中间设备当成死连接掐掉。**必须小于**你链路上最严格的空闲超时（nginx
  默认 60s）。

### 客户端断开 = 取消（不是暂停）

客户端断开连接时，服务端的行为是**彻底取消这一轮**：

1. 挂起的模型拉取被 `cancel()` 并等待落地；
2. agent 的生成器被 `aclose()`；
3. agent 的契约是「中途放弃不写长期记忆」—— 半途而废的问答不会污染会话记忆；
4. 本轮的对话**不会** `record` 进会话（只有收到 `agent.result` 才记录）。

推论：**断线后这一轮作废，客户端应当整轮重发**。重发时如果上一轮的取消
还没落地（毫秒级窗口），会得到 409，稍候重试即可。

### 不支持断线续传（以及为什么）

SSE 协议原生支持 `id:` 字段 + 客户端重连时带 `Last-Event-ID` 实现续传。
本项目**刻意没做**：

- 续传的前提是事件有可持久化的序号，而当前会话状态在进程内存里——
  重连可能打到另一个进程（多副本）或进程重启后的空存储，序号无从对账；
- 帧编码器（`sse.py`）预留了 `event_id` 参数，但服务层不用它；
  没有半吊子的「看起来能续传」。

方向：v0.7 会话持久化（Postgres `runs` 表按幂等键存结果）落地后，重连的
正确形态是「幂等重放」而不是「帧序号续传」—— 详见
`learning/v0.7-session-store-design.md`（设计文档，未实现）。

### 同会话并发：409，不排队

同一会话上一轮没结束就发下一轮，立即返回 409。这是有意的背压设计：
排队没有上限就是内存放大器（一个人狂点发送，服务替他攒几百个模型调用）。
客户端拿到 409 的正确处理是：等 `agent.result` 或 `error` 帧之后再发。

### 浏览器客户端注意

SSE 用 POST（请求体 + 鉴权头 + 幂等键都要用），所以浏览器原生
`EventSource` 不可用，用 `fetch` + ReadableStream 自己解析帧
（`data:` 行以 `\n\n` 分帧，多条 `data:` 行客户端拼回一个字符串）。

---

## 2. 指标：PromQL 速查

`/metrics` 输出 Prometheus 文本格式（`text/plain; version=0.0.4`）。
三个指标，标签一律用路由模板（势有界）：

| 指标 | 类型 | 标签 | 回答的问题 |
| --- | --- | --- | --- |
| `nexus_http_requests_total` | counter | `route`, `status` | 谁在多大比例上失败 |
| `nexus_http_request_duration_seconds` | histogram | `route` | 慢在哪个路由、慢到什么程度 |
| `nexus_http_requests_in_flight` | gauge | — | 此刻挂着多少请求（含 SSE 长连接） |

`/healthz` 与 `/metrics` 自身的流量不计入（探针会稀释真实信号）。
**单进程计数**：多副本部署时靠抓取侧 `sum by (instance)` 聚合。

### 每路由 P99 延迟

```promql
histogram_quantile(
  0.99,
  sum by (route, le) (rate(nexus_http_request_duration_seconds_bucket[5m]))
)
```

注意：SSE 路由的时长记到**流结束**（不是响应头发出时），所以
`/sessions/{session_id}/messages/stream` 的 P99 = 一整轮对话的时长，
和同步 JSON 路由的数字不是一个量级——这是特性，不是 bug。

### 错误率（错误预算）

```promql
sum by (route) (rate(nexus_http_requests_total{status=~"5.."}[5m]))
  / sum by (route) (rate(nexus_http_requests_total[5m]))
```

### 限流命中（429）

429 在路由匹配**之前**就被限流层短路，标签落在 `route="<unmatched>"`：

```promql
sum (rate(nexus_http_requests_total{status="429"}[5m]))
```

429 计入 `requests_total`（指标层在限流层外），也进访问日志（日志层在
限流层外）—— 两处都能查账。

### 背压告警

```promql
nexus_http_requests_in_flight > 100      # 阈值按 max_sessions 与机型调整
```

in-flight 持续爬升而不回落 = 消费者比生产者慢（或模型持续变慢），
是过载的第一个信号，早于错误率。

---

## 3. 日志：字段契约与排障路径

`NEXUS_LOG_JSON=true` 开启 JSON Lines（容器里推荐）。一行一个对象：

```json
{"ts":"...","level":"info","logger":"nexus_agents.service.access",
 "event":"POST /sessions -> 201 in 1.2ms","request_id":"9f86d081884c47d4",
 "method":"POST","path":"/sessions","status":201,"duration_ms":1.2}
```

业务日志（provider 失败、流中断）带同样的 `request_id` 和 `session_id`；
异常折进 `exception` 字段（永远一行一条）。`/healthz` 不记访问日志。

**排障的标准路径**：用户报障 → 让他给响应头里的 `X-Request-ID` →

```bash
docker compose logs service | grep 9f86d081884c47d4
# 或 JSON 管道：{request_id="9f86d081884c47d4"}
```

客户端没带 `X-Request-ID` 时服务端自动生成（16 hex）；带了就沿用
（多服务链路可串成一条）。带换行或超长（>128）的非法 ID 会被换新
—— 防日志注入。

---

## 4. 环境变量参考

全部变量都有默认值，只有 hosted 端点的密钥是硬要求。**空字符串等于没
设置**（compose 里 `NEXUS_API_KEY:` 留空 = 未设置）。拼错的布尔值
（如 `tur`）启动即报错退出（退出码 2），不会静默变 true。

### 必填（hosted 模型）

| 变量 | 说明 |
| --- | --- |
| `NEXUS_API_KEY` | 模型密钥（可被 `OPENAI_API_KEY` 兜底）。设了 `NEXUS_BASE_URL` 指向本地无需密钥的服务时可免 |

### Agent

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NEXUS_MODEL` | `gpt-4o-mini` | 模型名 |
| `NEXUS_BASE_URL` | — | OpenAI 兼容端点（本地 vLLM/ollama 等） |
| `NEXUS_TEMPERATURE` | — | 可选浮点 |
| `NEXUS_MAX_TOKENS` | — | 可选整数 |
| `NEXUS_SYSTEM_PROMPT` | — | 覆盖默认 system |
| `NEXUS_MAX_STEPS` | `12` | 单轮最大推理步数 |

### 服务

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NEXUS_HOST` / `NEXUS_PORT` | `0.0.0.0` / `8000` | 监听地址 |
| `NEXUS_SERVICE_KEY` | — | 共享密钥（闸门，非鉴权；设置后请求须带 `X-API-Key`） |
| `NEXUS_MAX_INPUT_CHARS` | `8000` | 单条输入上限（超出 413） |
| `NEXUS_MAX_SESSIONS` | `1000` | 在线会话上限（超出 LRU 淘汰） |
| `NEXUS_SESSION_TTL` | `3600` | 会话空闲过期秒数（过期后 404） |
| `NEXUS_HISTORY_MESSAGES` | `40` | 每请求携带的逐字历史条数 |
| `NEXUS_HEARTBEAT` | `15` | SSE 心跳间隔秒数（须小于中间设备空闲超时） |

### 限流 / 日志 / 停机

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `NEXUS_RATE_LIMIT` | 关 | 每分钟每调用方请求数；不设 = 不限流 |
| `NEXUS_RATE_LIMIT_BURST` | `10` | 令牌桶容量（允许的瞬时突发） |
| `NEXUS_LOG_JSON` | `false` | JSON Lines 日志 |
| `NEXUS_LOG_LEVEL` | `info` | critical / error / warning / info / debug / trace |
| `NEXUS_GRACEFUL_TIMEOUT` | `10` | SIGTERM 后等待收尾的秒数（compose 的 stop_grace_period 应比它长） |

---

## 5. 常见故障速查

| 症状 | 含义 | 处置 |
| --- | --- | --- |
| 客户端 404「session not found or expired」 | 会话过期（TTL）或被 LRU 淘汰，或进程重启（内存存储清零） | 客户端重建会话重发；频繁出现 → 调大 `NEXUS_SESSION_TTL` 或等 v0.7 持久化 |
| 客户端 409 | 同会话上一轮未结束 | 等上一轮结束再发；不要自动重试轰炸 |
| 客户端 429 + `Retry-After` | 命中限流 | 按 `Retry-After` 等待；服务端调 `NEXUS_RATE_LIMIT` |
| 客户端 502 | 上游模型失败（非本服务 bug） | 查日志里该 request_id 的 `provider failed`；检查 `NEXUS_BASE_URL` / 密钥 |
| SSE 流卡住，只有 `: keep-alive` | 模型在思考（正常）或上游真的挂了（超时前） | 看时长与上游健康；心跳说明连接本身是活的 |
| SSE 流中适数帧后静默断开 | 客户端侧断开或中间设备掐线 | 本轮作废，整轮重发；缩短中间设备超时或调小 `NEXUS_HEARTBEAT` |
| `--check` 退出码 2 | 配置错误（缺密钥 / 数值越界 / 布尔拼错） | 按错误信息修 env；`--check` 不起服务、不碰网络 |
| `/healthz` 200 但业务 502 | 进程活着，上游模型挂了 | 活跃探针刻意不碰模型——这正常；模型健康要靠业务错误率指标 |
