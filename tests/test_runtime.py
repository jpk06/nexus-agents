"""Deployment-config tests — the layer that fails at 3am on someone else's machine.

为什么这些测试值得写
--------------------
配置错误是最没技术含量、却最容易在「本地好使、容器里崩」的一类故障。
它们全部可以用一个 ``dict`` 覆盖，不需要起进程、不需要网络、不需要 Docker。
既然成本这么低，就没有理由把它留给「部署时再说」。

三条被单独钉死的性质：

1. **启动即失败**：缺密钥/端口写错必须在解析阶段抛错，而不是等第一个请求。
2. **空字符串 == 未设置**：Compose 里 ``VAR:`` 展开成空串太常见了。
3. **密钥永不入日志**：``repr()`` 和 ``describe()`` 都不能带出密钥内容 ——
   这两条路径覆盖了 99% 的「不小心打印了配置」。
"""

from __future__ import annotations

import importlib.abc
import sys
import types

import httpx
import pytest

from nexus_agents.errors import MissingCredentialsError
from nexus_agents.service import (
    BUILTIN_TOOLS,
    EnvConfigError,
    RuntimeConfig,
    ServiceSettings,
    build_agent,
    create_app,
    current_time,
    parse_env,
)
from nexus_agents.service.__main__ import main

SECRET = "sk-do-not-leak-me"


def env(**overrides: str) -> dict[str, str]:
    """A minimal valid environment; ``overrides`` are applied on top."""
    base = {"NEXUS_API_KEY": SECRET}
    base.update(overrides)
    return base


# ================================================================ 默认值


def test_defaults_with_only_an_api_key() -> None:
    config = parse_env(env())

    assert config.model == "gpt-4o-mini"
    assert config.base_url is None
    assert config.api_key == SECRET
    assert config.host == "0.0.0.0"
    assert config.port == 8000
    assert config.log_level == "info"
    assert config.graceful_timeout == 10
    # ServiceSettings 自带默认值，这里只确认「没被环境变量意外改掉」
    assert config.service == ServiceSettings()


def test_blank_values_fall_back_to_defaults() -> None:
    """``NEXUS_MODEL=`` 在 Compose 里等价于「没填」，不该变成空模型名。"""
    config = parse_env(env(NEXUS_MODEL="   ", NEXUS_LOG_LEVEL="", NEXUS_HOST=""))

    assert config.model == "gpt-4o-mini"
    assert config.log_level == "info"
    assert config.host == "0.0.0.0"


def test_explicit_mapping_wins_over_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """传了 mapping 就只读 mapping：否则测试之间会互相污染。"""
    monkeypatch.setenv("NEXUS_MODEL", "from-the-process")
    assert parse_env({"NEXUS_API_KEY": SECRET, "NEXUS_MODEL": "explicit"}).model == "explicit"


# ================================================================ 凭据


def test_api_key_falls_back_to_openai_api_key() -> None:
    """已经设过 OPENAI_API_KEY 的人不用再配一遍。"""
    config = parse_env({"OPENAI_API_KEY": "sk-from-openai"})

    assert config.api_key == "sk-from-openai"


def test_nexus_api_key_wins_over_the_openai_one() -> None:
    config = parse_env({"NEXUS_API_KEY": "sk-nexus", "OPENAI_API_KEY": "sk-openai"})

    assert config.api_key == "sk-nexus"


def test_missing_key_on_a_hosted_endpoint_fails_fast() -> None:
    with pytest.raises(EnvConfigError) as excinfo:
        parse_env({})

    message = str(excinfo.value)
    # 报错必须自己说明白怎么修，而不是只说「缺少配置」
    assert "NEXUS_API_KEY" in message
    assert "NEXUS_BASE_URL" in message


def test_explicit_default_base_url_still_needs_a_key() -> None:
    """把默认端点显式写一遍，不该被当成「自建端点」而放过。"""
    with pytest.raises(EnvConfigError):
        parse_env({"NEXUS_BASE_URL": "https://api.openai.com/v1/"})


def test_missing_key_is_fine_for_a_local_endpoint() -> None:
    """本地 vLLM / Ollama 通常不校验密钥。"""
    config = parse_env({"NEXUS_BASE_URL": "http://localhost:11434/v1"})

    assert config.api_key is None


# ================================================================ 数值校验


def test_port_must_be_an_integer() -> None:
    with pytest.raises(EnvConfigError) as excinfo:
        parse_env(env(NEXUS_PORT="8000:8000"))

    assert "NEXUS_PORT" in str(excinfo.value)
    assert "8000:8000" in str(excinfo.value)


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_port_must_be_in_range(port: str) -> None:
    with pytest.raises(EnvConfigError, match="NEXUS_PORT"):
        parse_env(env(NEXUS_PORT=port))


def test_session_ttl_rejects_garbage() -> None:
    with pytest.raises(EnvConfigError, match="NEXUS_SESSION_TTL"):
        parse_env(env(NEXUS_SESSION_TTL="one hour"))


def test_session_ttl_must_be_positive() -> None:
    """0 秒 TTL 意味着每个会话一建就过期 —— 静默的灾难，宁可启动失败。"""
    with pytest.raises(EnvConfigError, match="NEXUS_SESSION_TTL"):
        parse_env(env(NEXUS_SESSION_TTL="0"))


def test_heartbeat_accepts_fractional_seconds() -> None:
    config = parse_env(env(NEXUS_HEARTBEAT="0.05"))

    assert config.service.heartbeat_seconds == pytest.approx(0.05)


def test_history_messages_may_be_zero() -> None:
    """0 是合法值（只靠记忆、不传逐字历史），不能被当成「没填」。"""
    assert parse_env(env(NEXUS_HISTORY_MESSAGES="0")).service.history_messages == 0


def test_log_level_is_normalised_and_validated() -> None:
    assert parse_env(env(NEXUS_LOG_LEVEL="INFO")).log_level == "info"
    with pytest.raises(EnvConfigError, match="NEXUS_LOG_LEVEL"):
        parse_env(env(NEXUS_LOG_LEVEL="verbose"))


def test_optional_sampling_knobs_are_parsed() -> None:
    config = parse_env(env(NEXUS_TEMPERATURE="0.2", NEXUS_MAX_TOKENS="512"))

    assert config.agent.temperature == pytest.approx(0.2)
    assert config.agent.max_tokens == 512
    # 不填就是 None —— 别偷偷塞一个默认值进请求体
    assert parse_env(env()).agent.temperature is None


def test_optional_knobs_reject_garbage() -> None:
    with pytest.raises(EnvConfigError, match="NEXUS_TEMPERATURE"):
        parse_env(env(NEXUS_TEMPERATURE="warm"))
    with pytest.raises(EnvConfigError, match="NEXUS_MAX_TOKENS"):
        parse_env(env(NEXUS_MAX_TOKENS="lots"))


def test_service_key_becomes_the_gate() -> None:
    config = parse_env(env(NEXUS_SERVICE_KEY="gate-key"))

    assert config.service.api_key == "gate-key"
    # 闸门密钥和模型密钥是两件事，别串在一起
    assert config.api_key == SECRET


# ================================================================ 密钥不外泄


def test_repr_never_contains_the_key() -> None:
    config = parse_env(env(NEXUS_SERVICE_KEY="gate-key"))

    assert SECRET not in repr(config)
    assert "gate-key" not in repr(config)
    assert "api_key" not in repr(config)


def test_describe_reports_presence_not_value() -> None:
    text = parse_env(env(NEXUS_SERVICE_KEY="gate-key")).describe()

    assert "api_key=set" in text
    assert "service_key=required" in text
    assert SECRET not in text
    assert "gate-key" not in text


def test_describe_shows_open_when_no_gate() -> None:
    assert "service_key=open" in parse_env(env()).describe()


# ================================================================ 装配


def test_build_agent_wires_model_and_config() -> None:
    config = parse_env(
        env(
            NEXUS_MODEL="qwen2.5-7b-instruct",
            NEXUS_BASE_URL="http://localhost:8001/v1",
            NEXUS_MAX_STEPS="5",
            NEXUS_SYSTEM_PROMPT="你是仓库管理员",
        )
    )
    agent = build_agent(config)

    assert agent.name == "nexus-service"
    assert agent.config.max_steps == 5
    assert agent.config.system_prompt == "你是仓库管理员"
    assert [spec.name for spec in agent.tools] == [spec.name for spec in BUILTIN_TOOLS]
    provider = agent.provider
    assert getattr(provider, "model", None) == "qwen2.5-7b-instruct"


def test_build_agent_accepts_a_custom_toolset() -> None:
    agent = build_agent(parse_env(env()), tools=())

    assert agent.tools == ()


def test_build_agent_without_a_key_raises_the_provider_error() -> None:
    """``parse_env`` 拦在前面的那道检查之外，provider 自己还有一层兜底。"""
    config = RuntimeConfig(
        agent=parse_env(env()).agent,
        service=ServiceSettings(),
        model="gpt-4o-mini",
        base_url=None,
        api_key=None,
    )
    with pytest.raises(MissingCredentialsError):
        build_agent(config)


@pytest.mark.asyncio
async def test_healthz_answers_from_the_assembled_app() -> None:
    """装配的最终产物必须真的能应答 —— 否则配置正确也没意义。"""
    config = parse_env(env(NEXUS_BASE_URL="http://localhost:9/v1", NEXUS_MODEL="test-model"))
    app = create_app(build_agent(config), settings=config.service)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://service"
    ) as client:
        response = await client.get("/healthz")

    assert response.status_code == 200
    payload = response.json()
    assert payload["agent"] == "nexus-service"
    assert payload["tools"] == ["current_time"]


# ================================================================ 入口


def test_check_mode_prints_config_and_exits_zero(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NEXUS_API_KEY", SECRET)
    monkeypatch.delenv("NEXUS_SERVICE_KEY", raising=False)

    assert main(["--check"]) == 0

    out = capsys.readouterr().out
    assert "model=gpt-4o-mini" in out
    assert SECRET not in out


def test_check_mode_fails_loudly_on_a_bad_environment(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("NEXUS_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("NEXUS_BASE_URL", raising=False)

    # 退出码 2 = 用法/配置错误，和「运行中失败(1)」区分开
    assert main(["--check"]) == 2

    err = capsys.readouterr().err
    assert "configuration error" in err
    assert "NEXUS_API_KEY" in err


class _FakeUvicorn(types.ModuleType):
    """Stand-in for uvicorn that records the call instead of binding a port."""

    def __init__(self) -> None:
        super().__init__("uvicorn")
        self.calls: list[dict[str, object]] = []

    def run(self, app: object, **kwargs: object) -> None:  # 与 uvicorn 的真实签名对齐
        self.calls.append({"app": app, **kwargs})


def test_serve_path_hands_the_config_to_uvicorn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配置真的传下去了吗 —— 这类「接线错误」只能靠断言调用参数发现。"""
    fake = _FakeUvicorn()
    monkeypatch.setitem(sys.modules, "uvicorn", fake)
    monkeypatch.setenv("NEXUS_API_KEY", SECRET)
    monkeypatch.setenv("NEXUS_HOST", "127.0.0.1")
    monkeypatch.setenv("NEXUS_PORT", "9123")
    monkeypatch.setenv("NEXUS_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("NEXUS_GRACEFUL_TIMEOUT", "7")

    assert main([]) == 0

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["host"] == "127.0.0.1"
    assert call["port"] == 9123
    assert call["log_level"] == "warning"
    assert call["timeout_graceful_shutdown"] == 7
    # app 是装配好的 ASGI 应用，而不是 agent 或 None
    assert hasattr(call["app"], "routes")


def test_serve_path_explains_the_missing_extra(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """只装了核心包的人不该看到一个裸 ImportError。"""
    monkeypatch.setenv("NEXUS_API_KEY", SECRET)
    monkeypatch.delitem(sys.modules, "uvicorn", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_NoUvicorn(), *sys.meta_path])

    assert main([]) == 2
    assert "nexus-agents[api]" in capsys.readouterr().err


class _NoUvicorn(importlib.abc.MetaPathFinder):
    """让 ``import uvicorn`` 真的抛 ModuleNotFoundError。

    只把 ``sys.modules["uvicorn"]`` 设成 ``None`` 是不够的：那种情况下
    import 抛的是 ``ImportError``，和真实的「没装这个包」不是一条路径。
    """

    def find_spec(
        self,
        fullname: str,
        path: object = None,  # type: ignore[assignment]
        target: object = None,
    ) -> None:
        if fullname == "uvicorn":
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None



# ================================================================ 内置工具


def test_current_time_returns_the_offset() -> None:
    text = current_time("beijing")

    assert text.endswith("(UTC+8)")
    assert len(text.split(" ")[0]) == len("2026-01-01")


def test_current_time_guesses_aliases_case_insensitively() -> None:
    assert current_time("TOKYO").endswith("(UTC+9)")


def test_current_time_lists_aliases_for_an_unknown_zone() -> None:
    """错误信息要能引导模型自我修正，而不是只说「失败」。"""
    text = current_time("mars")

    assert "unknown zone" in text
    assert "beijing" in text


def test_builtin_tool_schema_is_exposed() -> None:
    spec = BUILTIN_TOOLS[0]

    assert spec.name == "current_time"
    assert list(spec.parameters["properties"]) == ["zone"]
    assert spec.parameters["required"] == ["zone"]
