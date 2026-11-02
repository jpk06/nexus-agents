"""Tool definition, validation, execution and registry.

The contract of a tool in nexus-agents
--------------------------------------
1. A tool is described to the model by a JSON Schema derived from its
   signature (see :mod:`nexus_agents.schema`).
2. Arguments coming back from the model are *validated and coerced* before
   the Python function is ever called, so tools never see ``"3"`` where they
   declared ``int``.
3. A tool that raises does **not** crash the agent. Its exception becomes a
   ``ToolResult(is_error=True)`` that is fed back to the model, which can
   then retry with corrected arguments or pick another tool.
"""

from __future__ import annotations

import inspect
import json
import typing
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from .errors import DuplicateToolError, InvalidToolArguments, ToolNotFoundError
from .schema import build_parameters_schema, parse_docstring
from .types import ToolCall, ToolResult

__all__ = ["ToolSpec", "ToolRegistry", "tool", "build_tool_spec"]

_SELF_LIKE = {"self", "cls"}


def _pascal_case(name: str) -> str:
    parts = [p for p in name.replace("-", "_").split("_") if p]
    return "".join(p[:1].upper() + p[1:] for p in parts) or "Tool"


def _format_validation_error(error: ValidationError) -> str:
    """把 pydantic 的报错压成一行，省 token 也更容易让模型读懂。"""
    details = []
    for item in error.errors():
        location = ".".join(str(p) for p in item["loc"]) or "<root>"
        details.append(f"{location}: {item['msg']}")
    return "; ".join(details)


def stringify_result(value: Any) -> str:
    """Render any Python value as the string a model should see."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json(ensure_ascii=False)
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


@dataclass(frozen=True)
class ToolSpec:
    """An immutable, validated description of one callable tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]
    _validator: type[BaseModel] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "_validator", self._build_validator())

    # ------------------------------------------------------------ 调用

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Keep decorated functions directly callable (nice for tests)."""
        return self.fn(*args, **kwargs)

    # ------------------------------------------------------------ 校验

    def _build_validator(self) -> type[BaseModel]:
        """Create a pydantic model mirroring the function signature."""
        try:
            signature = inspect.signature(self.fn)
            hints = typing.get_type_hints(self.fn)
        except Exception:
            return create_model(_pascal_case(self.name) + "Args",
                                __config__=ConfigDict(extra="allow"))

        fields: dict[str, Any] = {}
        for index, (pname, param) in enumerate(signature.parameters.items()):
            if index == 0 and pname in _SELF_LIKE:
                continue
            if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                              inspect.Parameter.VAR_KEYWORD):
                continue
            annotation = hints.get(pname, param.annotation)
            if annotation is inspect.Parameter.empty:
                annotation = Any
            if param.default is inspect.Parameter.empty:
                fields[pname] = (annotation, ...)
            else:
                fields[pname] = (annotation, param.default)

        return create_model(
            _pascal_case(self.name) + "Args",
            __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
            **fields,
        )

    def validate_arguments(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        """Validate and coerce raw model arguments into Python kwargs."""
        validator = self._validator
        if validator is None:  # pragma: no cover - __post_init__ 总会设置
            return dict(arguments or {})
        try:
            validated = validator.model_validate(arguments or {})
        except ValidationError as exc:
            raise InvalidToolArguments(self.name, _format_validation_error(exc)) from exc
        # exclude_unset：没给的字段交给 Python 默认值处理，避免覆盖 sentinel
        return validated.model_dump(exclude_unset=True)

    # ------------------------------------------------------------ 执行

    def invoke(self, call: ToolCall | None = None, **arguments: Any) -> ToolResult:
        """Run the tool and always return a :class:`ToolResult`.

        任何失败都被「捕获成数据」返回，而不是向上抛 —— 这样 agent 循环
        永远不会被一个坏工具打断。
        """
        call = call or ToolCall(id="", name=self.name, arguments=arguments)
        try:
            kwargs = self.validate_arguments(call.arguments)
        except InvalidToolArguments as exc:
            return ToolResult.failure(call_id=call.id, name=self.name, error=str(exc))

        try:
            value = self.fn(**kwargs)
        except Exception as exc:
            return ToolResult.failure(
                call_id=call.id, name=self.name, error=f"{type(exc).__name__}: {exc}"
            )
        return ToolResult(
            tool_call_id=call.id, name=self.name, output=stringify_result(value)
        )

    # ------------------------------------------------------------ 对外格式

    def to_api_payload(self) -> dict[str, Any]:
        """OpenAI / vLLM / Ollama function-calling compatible definition."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def build_tool_spec(
    func: Callable[..., Any], *, name: str | None = None, description: str | None = None
) -> ToolSpec:
    """Build a :class:`ToolSpec` from a plain Python callable."""
    if isinstance(func, ToolSpec):
        return func
    if not callable(func):
        raise TypeError(f"{func!r} is not callable")

    summary, _ = parse_docstring(inspect.getdoc(func))
    resolved_name = name or str(getattr(func, "__name__", "tool"))
    return ToolSpec(
        name=resolved_name,
        description=description or summary or f"Call {resolved_name}.",
        parameters=build_parameters_schema(func),
        fn=func,
    )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorator turning a function into a :class:`ToolSpec`.

    Supports both ``@tool`` and ``@tool(name="...", description="...")``.
    The decorated object stays callable, so existing unit tests keep working.
    """
    if func is not None:
        return build_tool_spec(func, name=name, description=description)

    def decorator(target: Callable[..., Any]) -> ToolSpec:
        return build_tool_spec(target, name=name, description=description)

    return decorator


class ToolRegistry:
    """An ordered, name-unique collection of tools.

    注册中心是 agent 唯一能碰到工具的入口，这样「模型能调用什么」这件事
    有单一事实来源，也便于按权限裁剪工具集。
    """

    def __init__(self, tools: Iterable[ToolSpec | Callable[..., Any]] = ()) -> None:
        self._tools: dict[str, ToolSpec] = {}
        for item in tools:
            self.register(item)

    # ------------------------------------------------------------ 增删

    def register(
        self,
        item: ToolSpec | Callable[..., Any],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        spec = item if isinstance(item, ToolSpec) else build_tool_spec(
            item, name=name, description=description
        )
        if spec.name in self._tools:
            raise DuplicateToolError(spec.name)
        self._tools[spec.name] = spec
        return spec

    # ------------------------------------------------------------ 查询

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools.values())

    def get(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(name, tuple(self._tools)) from None

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        return f"ToolRegistry({', '.join(self._tools) or 'empty'})"

    # ------------------------------------------------------------ 组合

    def with_tools(self, *tools: ToolSpec | Callable[..., Any]) -> ToolRegistry:
        """Return a *new* registry containing the current tools plus more."""
        return ToolRegistry((*self.specs(), *tools))

    def restricted_to(self, names: Iterable[str]) -> ToolRegistry:
        """Return a new registry limited to the given tool names.

        多智能体里给不同角色发不同工具集时非常有用。
        """
        wanted = set(names)
        missing = wanted - set(self._tools)
        if missing:
            raise ToolNotFoundError(sorted(missing)[0], tuple(self._tools))
        return ToolRegistry([self._tools[n] for n in self._tools if n in wanted])

    # ------------------------------------------------------------ 执行

    def invoke(self, call: ToolCall) -> ToolResult:
        """Dispatch one model-requested tool call."""
        try:
            spec = self.get(call.name)
        except ToolNotFoundError as exc:
            return ToolResult.failure(call_id=call.id, name=call.name, error=str(exc))
        return spec.invoke(call)

    def to_api_payload(self) -> list[dict[str, Any]]:
        return [spec.to_api_payload() for spec in self._tools.values()]
