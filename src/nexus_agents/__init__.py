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
from .checkpoint import Checkpoint, Checkpointer, InMemoryCheckpointer
from .errors import (
    AgentError,
    CheckpointError,
    DuplicateNodeError,
    DuplicateToolError,
    GraphError,
    GraphStepLimitExceeded,
    InvalidRouteError,
    InvalidToolArguments,
    MaxStepsExceeded,
    MissingCredentialsError,
    MissingEntryPointError,
    NexusError,
    NodeNotFoundError,
    ProviderError,
    ProviderResponseError,
    ProviderTransportError,
    SchemaError,
    ThreadNotFoundError,
    ToolError,
    ToolNotFoundError,
    UnreachableNodeError,
)
from .graph import (
    END,
    CompiledGraph,
    GraphResult,
    GraphStatus,
    StateGraph,
    append,
    apply_update,
    keep_first,
    max_value,
    merge,
    replace,
    sum_values,
)
from .memory import (
    SUMMARY_PREFIX,
    ContextPolicy,
    ContextResult,
    ContextStats,
    ConversationMemory,
    ExtractiveSummarizer,
    HeuristicTokenEstimator,
    InMemoryMemoryStore,
    LLMSummarizer,
    MemoryRecord,
    MemoryStore,
    Summarizer,
    TokenEstimator,
    estimate_message_tokens,
    render_transcript,
    tokenize,
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

__version__ = "0.3.0"

__all__ = [
    # agent
    "AgentConfig",
    "AgentResult",
    "ReActAgent",
    "Stopped",
    # graph
    "END",
    "StateGraph",
    "CompiledGraph",
    "GraphResult",
    "GraphStatus",
    "apply_update",
    "append",
    "merge",
    "replace",
    "keep_first",
    "sum_values",
    "max_value",
    # checkpoint
    "Checkpoint",
    "Checkpointer",
    "InMemoryCheckpointer",
    # memory
    "SUMMARY_PREFIX",
    "ContextPolicy",
    "ContextResult",
    "ContextStats",
    "ConversationMemory",
    "ExtractiveSummarizer",
    "HeuristicTokenEstimator",
    "InMemoryMemoryStore",
    "LLMSummarizer",
    "MemoryRecord",
    "MemoryStore",
    "Summarizer",
    "TokenEstimator",
    "estimate_message_tokens",
    "render_transcript",
    "tokenize",
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
    # graph errors
    "GraphError",
    "DuplicateNodeError",
    "NodeNotFoundError",
    "MissingEntryPointError",
    "UnreachableNodeError",
    "InvalidRouteError",
    "GraphStepLimitExceeded",
    "CheckpointError",
    "ThreadNotFoundError",
    "__version__",
]
