"""Core message primitives shared by every agent, provider and tool.

这里刻意只用标准库 + pydantic：消息对象是整个框架的「通用货币」，
任何一层都要能读写它，所以它必须简单、不可变、可序列化。
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Role",
    "ToolCall",
    "ToolResult",
    "Usage",
    "Message",
    "FinishReason",
    "system_message",
    "user_message",
    "assistant_message",
    "tool_result_message",
]


class Role(str, Enum):
    """Conversation roles understood by OpenAI-compatible chat APIs."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(str, Enum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    ERROR = "error"


class ToolCall(BaseModel):
    """A single tool invocation requested by the model."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    #: 模型给出的原始参数，尚未按工具的 schema 校验
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("arguments", mode="before")
    @classmethod
    def _decode_json_string(cls, value: Any) -> Any:
        # 很多模型会把 arguments 传成 JSON 字符串而不是对象，这里统一兜住
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        if value is None:
            return {}
        return value


class ToolResult(BaseModel):
    """Outcome of running one tool call.

    ``is_error=True`` 时结果仍然会被送回模型 —— 这是 agent 能自我纠错的前提。
    """

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    output: str = ""
    error: str | None = None
    is_error: bool = False

    @classmethod
    def failure(cls, *, call_id: str, name: str, error: str) -> ToolResult:
        return cls(tool_call_id=call_id, name=name, error=error, is_error=True)


class Usage(BaseModel):
    """Token accounting for a single request/response pair."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )


class Message(BaseModel):
    """One immutable message in a conversation."""

    model_config = ConfigDict(frozen=True)

    role: Role
    content: str = ""
    #: 多智能体场景下的发言者名字；role=tool 时是工具名
    name: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    #: role=tool 的结果是否为错误
    is_error: bool = False

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    # ------------------------------------------------------------ 序列化

    def to_api_payload(self) -> dict[str, Any]:
        """Convert to the OpenAI chat-completions wire format.

        这样 provider 层不用各写一套转换逻辑，换后端只改 transport。
        """
        payload: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name is not None:
            payload["name"] = self.name
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        # arguments 在线上格式里是 JSON 字符串
                        "arguments": json.dumps(call.arguments, ensure_ascii=False),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id is not None:
            payload["tool_call_id"] = self.tool_call_id
        return payload


# ---------------------------------------------------------------- 构造器


def system_message(content: str) -> Message:
    return Message(role=Role.SYSTEM, content=content)


def user_message(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def assistant_message(
    content: str = "", *, tool_calls: tuple[ToolCall, ...] = ()
) -> Message:
    return Message(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)


def tool_result_message(result: ToolResult) -> Message:
    """Wrap a :class:`ToolResult` into the message the model expects back."""
    return Message(
        role=Role.TOOL,
        content=(result.error or "") if result.is_error else result.output,
        name=result.name,
        tool_call_id=result.tool_call_id,
        is_error=result.is_error,
    )
