"""nexus-agents — a small, explicit and testable runtime for LLM agents.

Motivation
----------
Frameworks hide the interesting parts. This package keeps every piece of an
agent — messages, tool schemas, provider transport, the control loop — small
enough to read in one sitting, while staying production-shaped: typed,
validated, traced and fully unit-testable without network access.

Quick start
-----------
>>> from nexus_agents import ReActAgent, tool
>>> from nexus_agents.providers import ScriptedProvider, tool_call_message
>>>
>>> @tool
... def add(a: int, b: int) -> int:
...     '''Add two numbers.'''
...     return a + b
>>>
>>> provider = ScriptedProvider([tool_call_message("add", {"a": 2, "b": 3}), "5"])
>>> result = ReActAgent(provider, [add]).run("What is 2 + 3?")
>>> result.answer
'5'
"""

from __future__ import annotations

from .agent import AgentConfig, AgentResult, ReActAgent, Stopped
from .errors import (
    AgentError,
    DuplicateToolError,
    InvalidToolArguments,
    MaxStepsExceeded,
    MissingCredentialsError,
    NexusError,
    ProviderError,
    ProviderResponseError,
    ProviderTransportError,
    SchemaError,
    ToolError,
    ToolNotFoundError,
)
from .schema import annotation_to_json_schema, build_parameters_schema, parse_docstring
from .tools import ToolRegistry, ToolSpec, build_tool_spec, tool
from .tracing import EventType, TraceEvent, Tracer
from .types import (
    FinishReason,
    Message,
    Role,
    ToolCall,
    ToolResult,
    Usage,
    assistant_message,
    system_message,
    tool_result_message,
    user_message,
)

__version__ = "0.1.0"

__all__ = [
    # agent
    "AgentConfig",
    "AgentResult",
    "ReActAgent",
    "Stopped",
    # tools
    "ToolRegistry",
    "ToolSpec",
    "build_tool_spec",
    "tool",
    # schema
    "annotation_to_json_schema",
    "build_parameters_schema",
    "parse_docstring",
    # types
    "FinishReason",
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "Usage",
    "assistant_message",
    "system_message",
    "tool_result_message",
    "user_message",
    # tracing
    "EventType",
    "TraceEvent",
    "Tracer",
    # errors
    "NexusError",
    "ToolError",
    "ToolNotFoundError",
    "DuplicateToolError",
    "InvalidToolArguments",
    "SchemaError",
    "ProviderError",
    "ProviderTransportError",
    "ProviderResponseError",
    "MissingCredentialsError",
    "AgentError",
    "MaxStepsExceeded",
    "__version__",
]
