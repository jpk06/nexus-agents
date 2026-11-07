"""Environment-driven assembly: from 20 environment variables to a runnable app.

为什么把「解析环境变量」和「装配对象」拆成两个函数
--------------------------------------------------
``parse_env`` 是纯函数：给一个 ``Mapping[str, str]``，返回一个不可变的
``RuntimeConfig``。它不建 httpx 客户端、不碰 FastAPI、不读全局 ``os.environ``
（只在没传参时才读）。这样部署配置这件事就能被**单元测试**覆盖 —— 而部署
配置恰恰是最容易在「本地好使、容器里崩」的地方出错的那一层。

``build_agent`` 才负责把配置变成对象。两件事分开，测试就不用为了验证
「NEXUS_PORT 写错了会不会报错」而去起一个进程。

三条启动期规则
--------------
1. **启动即失败（fail fast）**。配置错了就在 1 秒内退出并说清哪一项错了，
   而不是等第一个用户请求进来才 500。容器编排器（Compose / K8s）能把
   「进程立刻退出并打印原因」变成一次明确的部署失败 —— 这是最好的结果。
2. **空字符串等于没设置**。``docker-compose.yml`` 里 ``NEXUS_API_KEY:``
   展开成空串是极常见的事，把它当成「用户填了一个空密钥」只会制造噪音。
3. **密钥永远不进入任何输出**。``RuntimeConfig.api_key`` 标记了
   ``repr=False``，``describe()`` 只报「设置没设置」。日志是会被复制粘贴到
   issue 里去的，密钥一旦进去就再也收不回来。
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

from ..agent import AgentConfig, ReActAgent
from ..providers.openai_compat import DEFAULT_BASE_URL, OpenAICompatibleProvider
from ..tools import ToolSpec
from .app import ServiceSettings
from .builtin import BUILTIN_TOOLS

__all__ = ["EnvConfigError", "RuntimeConfig", "parse_env", "build_agent"]

#: 默认模型。选一个小而便宜的：这个仓库演示的是运行时，不是模型能力。
DEFAULT_MODEL = "gpt-4o-mini"

#: uvicorn 认可的日志级别。算白名单而不是直接透传 —— 写错时 uvicorn 会
#: 在启动几秒后抛一个跟配置毫无关系的异常，不如在这里就拦住。
_LOG_LEVELS = ("critical", "error", "warning", "info", "debug", "trace")


class EnvConfigError(ValueError):
    """The environment cannot produce a runnable service.

    为什么不是 :class:`~nexus_agents.errors.NexusError`：这类错误发生在
    **启动期**，还没有任何 agent 在跑。混进 ``NexusError`` 家族会让上层
    用同一个 ``except`` 抓住「配置错了」和「运行失败了」，而这两件事的处置
    完全不同 —— 一个要改部署，一个要改代码或重试。
    """


@dataclass(frozen=True)
class RuntimeConfig:
    """Everything the process needs, resolved and validated."""

    agent: AgentConfig
    service: ServiceSettings
    model: str = DEFAULT_MODEL
    base_url: str | None = None
    #: ``repr=False``：异常回溯、调试打印、``dataclasses.asdict`` 之外的
    #: 一切「随手一 print」都不会把密钥带出去。
    api_key: str | None = field(default=None, repr=False)
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"
    #: 收到 SIGTERM 后等待在途请求收尾的秒数。SSE 连接可能正开到一半，
    #: 直接掐断会让客户端看到一个「没有 error 帧的截断流」。
    graceful_timeout: int = 10

    def describe(self) -> str:
        """One line per setting, safe to print in logs and CI output."""
        return "\n".join(
            [
                f"model={self.model}",
                f"base_url={self.base_url or DEFAULT_BASE_URL}",
                f"api_key={'set' if self.api_key else 'not set'}",
                f"system_prompt={'set' if self.agent.system_prompt else 'not set'}",
                f"max_steps={self.agent.max_steps}",
                f"max_input_chars={self.service.max_input_chars}",
                f"max_sessions={self.service.max_sessions}",
                f"session_ttl_seconds={self.service.session_ttl_seconds}",
                f"heartbeat_seconds={self.service.heartbeat_seconds}",
                f"history_messages={self.service.history_messages}",
                f"service_key={'required' if self.service.api_key else 'open'}",
                f"host={self.host}",
                f"port={self.port}",
                f"log_level={self.log_level}",
                f"graceful_timeout={self.graceful_timeout}",
            ]
        )


# ================================================================ 读取原语


def _text(environ: Mapping[str, str], name: str, default: str | None = None) -> str | None:
    raw = environ.get(name)
    if raw is None:
        return default
    raw = raw.strip()
    return raw or default


def _int(
    environ: Mapping[str, str],
    name: str,
    default: int,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    raw = _text(environ, name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        # 报错一定要带上变量名和原值：容器日志里只有这一行能告诉你哪里错了
        raise EnvConfigError(f"{name} must be an integer, got {raw!r}") from None
    if minimum is not None and value < minimum:
        raise EnvConfigError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise EnvConfigError(f"{name} must be <= {maximum}, got {value}")
    return value


def _float(
    environ: Mapping[str, str],
    name: str,
    default: float,
    *,
    exclusive_minimum: float | None = None,
) -> float:
    raw = _text(environ, name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise EnvConfigError(f"{name} must be a number, got {raw!r}") from None
    if exclusive_minimum is not None and value <= exclusive_minimum:
        raise EnvConfigError(f"{name} must be > {exclusive_minimum}, got {value}")
    return value


def _optional_int(environ: Mapping[str, str], name: str) -> int | None:
    raw = _text(environ, name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        raise EnvConfigError(f"{name} must be an integer, got {raw!r}") from None


def _optional_float(environ: Mapping[str, str], name: str) -> float | None:
    raw = _text(environ, name)
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        raise EnvConfigError(f"{name} must be a number, got {raw!r}") from None


# ================================================================ 解析


def parse_env(environ: Mapping[str, str] | None = None) -> RuntimeConfig:
    """Turn the environment into a validated :class:`RuntimeConfig`.

    ``environ`` 显式传入时不读 ``os.environ`` —— 测试需要能构造任意环境，
    而不是 ``monkeypatch`` 一堆全局状态再复原。
    """
    env = os.environ if environ is None else environ

    model = _text(env, "NEXUS_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    base_url = _text(env, "NEXUS_BASE_URL")
    # 允许复用生态里最通用的变量名：部署过 OpenAI SDK 的人已经设了它
    api_key = _text(env, "NEXUS_API_KEY") or _text(env, "OPENAI_API_KEY")

    # 提前判一次「有没有凭据」，和 provider 自己的检查**故意重复**：
    # provider 的报错发生在构造函数里，信息只提到变量名；这里能在进程启动
    # 前给出「该设哪个变量、或者把 base_url 指向本地服务」的完整指引。
    if api_key is None and (base_url is None or base_url.rstrip("/") == DEFAULT_BASE_URL):
        raise EnvConfigError(
            "no API key: set NEXUS_API_KEY (or OPENAI_API_KEY) for a hosted endpoint, "
            "or set NEXUS_BASE_URL to a local OpenAI-compatible server that needs none"
        )

    log_level = (_text(env, "NEXUS_LOG_LEVEL", "info") or "info").lower()
    if log_level not in _LOG_LEVELS:
        raise EnvConfigError(
            f"NEXUS_LOG_LEVEL must be one of {', '.join(_LOG_LEVELS)}, got {log_level!r}"
        )

    service = ServiceSettings(
        max_input_chars=_int(env, "NEXUS_MAX_INPUT_CHARS", 8000, minimum=1),
        max_sessions=_int(env, "NEXUS_MAX_SESSIONS", 1000, minimum=1),
        session_ttl_seconds=_float(env, "NEXUS_SESSION_TTL", 3600.0, exclusive_minimum=0.0),
        # 心跳必须明显小于中间设备（负载均衡、反代）的空闲超时，否则连接
        # 会在心跳到达之前就被对端掐掉。上限 60s 是常见默认值，超过就没意义了。
        heartbeat_seconds=_float(env, "NEXUS_HEARTBEAT", 15.0, exclusive_minimum=0.0),
        history_messages=_int(env, "NEXUS_HISTORY_MESSAGES", 40, minimum=0),
        api_key=_text(env, "NEXUS_SERVICE_KEY"),
    )

    agent = AgentConfig(
        max_steps=_int(env, "NEXUS_MAX_STEPS", 12, minimum=1),
        system_prompt=_text(env, "NEXUS_SYSTEM_PROMPT"),
        temperature=_optional_float(env, "NEXUS_TEMPERATURE"),
        max_tokens=_optional_int(env, "NEXUS_MAX_TOKENS"),
    )

    return RuntimeConfig(
        agent=agent,
        service=service,
        model=model,
        base_url=base_url,
        api_key=api_key,
        host=_text(env, "NEXUS_HOST", "0.0.0.0") or "0.0.0.0",
        # 上限卡在 65535：端口号写错（比如顺手填了 8080:8000 里的映射串）
        # 时，报错应该发生在解析阶段，而不是 bind() 的时候
        port=_int(env, "NEXUS_PORT", 8000, minimum=1, maximum=65535),
        log_level=log_level,
        graceful_timeout=_int(env, "NEXUS_GRACEFUL_TIMEOUT", 10, minimum=0),
    )


# ================================================================ 装配


def build_agent(
    config: RuntimeConfig,
    tools: Iterable[ToolSpec] = BUILTIN_TOOLS,
) -> ReActAgent:
    """Wire a provider and a tool set into one agent.

    agent 本身不持有会话状态（见 ``ReActAgent.run``），所以同一个实例可以
    同时服务所有会话 —— 单实例、单进程、无状态，正好可以被容器横向扩展。
    """
    provider = OpenAICompatibleProvider(
        model=config.model,
        base_url=config.base_url,
        api_key=config.api_key,
    )
    return ReActAgent(provider, tools, config=config.agent, name="nexus-service")
