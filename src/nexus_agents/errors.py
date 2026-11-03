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


# ---------------------------------------------------------------- Graph 层


class GraphError(NexusError):
    """Raised for invalid graph definitions and failed graph runs."""

    default_message = "graph error"


class DuplicateNodeError(GraphError):
    def __init__(self, name: str) -> None:
        super().__init__(f"Node {name!r} is already defined in this graph.")
        self.name = name


class NodeNotFoundError(GraphError):
    def __init__(self, name: str, available: tuple[str, ...] = ()) -> None:
        hint = f" Known nodes: {', '.join(available)}." if available else ""
        super().__init__(f"Unknown node {name!r}.{hint}")
        self.name = name
        self.available = available


class MissingEntryPointError(GraphError):
    default_message = "Graph has no entry point; call set_entry_point() first."


class UnreachableNodeError(GraphError):
    """A node exists but no edge or branch can ever reach it."""

    def __init__(self, names: tuple[str, ...]) -> None:
        super().__init__(
            "Unreachable node(s): "
            + ", ".join(names)
            + ". Add an edge or remove them."
        )
        self.names = names


class InvalidRouteError(GraphError):
    """A router returned a branch that the graph does not know."""

    def __init__(self, source: str, branch: str, allowed: tuple[str, ...]) -> None:
        super().__init__(
            f"Router for node {source!r} returned {branch!r}, "
            f"which is not a known node. Allowed: {', '.join(allowed)}."
        )
        self.source = source
        self.branch = branch
        self.allowed = allowed


class GraphStepLimitExceeded(GraphError):
    def __init__(self, max_steps: int, path: tuple[str, ...] = ()) -> None:
        tail = " -> ".join(path[-5:]) or "<no node ran>"
        super().__init__(
            f"Graph did not finish within {max_steps} steps (last nodes: {tail})."
        )
        self.max_steps = max_steps
        self.path = path


# ---------------------------------------------------------------- Checkpoint 层


class CheckpointError(GraphError):
    """Raised when a run cannot be resumed from stored state."""

    default_message = "checkpoint error"


class ThreadNotFoundError(CheckpointError):
    def __init__(self, thread_id: str) -> None:
        super().__init__(f"No checkpoint stored for thread {thread_id!r}.")
        self.thread_id = thread_id
