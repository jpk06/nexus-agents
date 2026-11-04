"""Exception hierarchy for nexus-agents.

设计原则：
- 所有异常都继承自 ``NexusError``，调用方可以一次性捕获。
- 「工具执行出错」不算致命错误，会被转成 ``ToolResult(is_error=True)``
  回喂给模型，由模型决定如何补救（这是 agent 工程的关键实践）。
- 只有配置错误、协议错误这类无法在运行期恢复的问题才真正抛出。
"""

from __future__ import annotations


class NexusError(Exception):
    """Base class for every error raised by nexus-agents."""

    default_message = "nexus-agents error"

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.default_message)
        self.message = message or self.default_message


# ---------------------------------------------------------------- 工具层


class ToolError(NexusError):
    """Raised when a tool cannot be executed at all (e.g. unknown tool)."""

    default_message = "tool execution failed"


class ToolNotFoundError(ToolError):
    def __init__(self, name: str, available: tuple[str, ...] = ()) -> None:
        hint = f" Available tools: {', '.join(available)}." if available else ""
        super().__init__(f"Unknown tool {name!r}.{hint}")
        self.name = name
        self.available = available


class DuplicateToolError(ToolError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Tool {name!r} is already registered.")
        self.name = name


class InvalidToolArguments(ToolError):
    """Raised when arguments do not satisfy the tool's declared schema."""

    def __init__(self, tool_name: str, detail: str) -> None:
        super().__init__(f"Invalid arguments for tool {tool_name!r}: {detail}")
        self.tool_name = tool_name
        self.detail = detail


# ---------------------------------------------------------------- Schema 层


class SchemaError(NexusError):
    """Raised when a Python callable cannot be turned into a JSON Schema."""

    default_message = "failed to build a tool schema"


class UnsupportedAnnotationError(SchemaError):
    def __init__(self, annotation: object) -> None:
        super().__init__(f"Unsupported type annotation: {annotation!r}")
        self.annotation = annotation


# ---------------------------------------------------------------- Provider 层


class ProviderError(NexusError):
    """Raised when an LLM backend returns an unusable response."""

    default_message = "LLM provider error"


class ProviderTransportError(ProviderError):
    """Network / HTTP level failure talking to the backend."""


class ProviderResponseError(ProviderError):
    """Backend replied, but the payload could not be parsed."""


class MissingCredentialsError(ProviderError):
    def __init__(self, env_var: str) -> None:
        super().__init__(
            f"Missing API credentials: set the {env_var} environment variable."
        )
        self.env_var = env_var


# ---------------------------------------------------------------- Agent 层


class AgentError(NexusError):
    """Raised when the agent loop cannot continue."""

    default_message = "agent error"


class MaxStepsExceeded(AgentError):
    def __init__(self, max_steps: int) -> None:
        super().__init__(f"Agent did not finish within {max_steps} steps.")
        self.max_steps = max_steps
