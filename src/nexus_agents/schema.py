"""Turn Python callables into JSON Schema tool definitions.

Why this exists
---------------
Function-calling APIs need a JSON Schema for every tool. Hand-writing those
schemas is how tool definitions drift out of sync with the actual code.
Here the schema is *derived* from the signature and docstring, so the
description a model sees can never disagree with the function it calls.

Supported annotations
---------------------
``str`` / ``int`` / ``float`` / ``bool``, ``list[T]``, ``dict[K, V]``,
``tuple[T, ...]``, ``set[T]``, ``Enum``, ``Literal``, ``Optional[T]``,
``X | Y``, pydantic models, and anything JSON-Schema-able pydantic supports.
"""

from __future__ import annotations

import datetime as _dt
import enum
import inspect
import re
import types
import typing
import uuid as _uuid
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel

from .errors import SchemaError

__all__ = [
    "annotation_to_json_schema",
    "parse_docstring",
    "build_parameters_schema",
]

#: 需要跳过的前导参数（装饰器用在类的方法上时会看到它们）
_SELF_LIKE = {"self", "cls"}

_SECTION_ALIASES = ("args", "arguments", "parameters", "params", "keyword args")

#: 参数段之后常见的其它段头 —— 必须识别出来，否则 "Returns:" 会被当成参数名
_OTHER_SECTIONS = frozenset({
    "returns", "return", "raises", "raise", "yields", "yield", "receives",
    "examples", "example", "notes", "note", "see also", "references",
    "attributes", "methods", "warns", "deprecated", "other parameters",
})

_PARAM_NAME = r"(?P<name>\*{0,2}[A-Za-z_]\w*)"

# Google 风格：  name (type): description
_GOOGLE_PARAM_RE = re.compile(
    _PARAM_NAME + r"\s*(?:\([^)]*\))?\s*:\s*(?P<desc>.*)$"
)
# numpy 风格：  name : type
_NUMPY_PARAM_RE = re.compile(_PARAM_NAME + r"\s*(?::\s*(?P<type>.*))?$")


def annotation_to_json_schema(annotation: Any) -> dict[str, Any]:
    """Convert a single Python annotation to a JSON Schema fragment.

    无法识别的注解返回 ``{}``（表示「任意值」）而不是报错 —— 让一个
    冷门类型毁掉整个工具注册不值得。
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {}

    if annotation is None or annotation is type(None):
        return {"type": "null"}

    # bool 必须在 int 之前判断：bool 是 int 的子类
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is str:
        return {"type": "string"}
    if annotation is bytes:
        return {"type": "string", "contentEncoding": "base64"}

    origin = get_origin(annotation)

    # --- X | Y 和 Union[X, Y] -------------------------------------------
    if origin is Union or isinstance(annotation, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if not args:
            return {"type": "null"}
        if len(args) == 1:
            return annotation_to_json_schema(args[0])
        return {"anyOf": [annotation_to_json_schema(a) for a in args]}

    # --- Literal[...] ----------------------------------------------------
    if origin is typing.Literal:
        values = list(get_args(annotation))
        schema: dict[str, Any] = {"enum": values}
        json_types = sorted({_json_type_name(type(v)) for v in values})
        if len(json_types) == 1:
            schema["type"] = json_types[0]
        return schema

    # --- Enum ------------------------------------------------------------
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        values = [m.value for m in annotation]
        schema = {"enum": values}
        json_types = sorted({_json_type_name(type(v)) for v in values})
        if len(json_types) == 1:
            schema["type"] = json_types[0]
        return schema

    # --- pydantic model --------------------------------------------------
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return _strip_pydantic_internals(annotation.model_json_schema())

    # --- 容器 -------------------------------------------------------------
    if origin in (list, list):
        type_args = get_args(annotation)
        schema = {"type": "array"}
        if type_args:
            schema["items"] = annotation_to_json_schema(type_args[0])
        return schema

    if origin in (set, frozenset, set, frozenset):
        type_args = get_args(annotation)
        schema = {"type": "array", "uniqueItems": True}
        if type_args:
            schema["items"] = annotation_to_json_schema(type_args[0])
        return schema

    if origin in (tuple, tuple):
        type_args = get_args(annotation)
        schema = {"type": "array"}
        if type_args and type_args[-1] is Ellipsis:
            schema["items"] = annotation_to_json_schema(type_args[0])
        elif type_args:
            schema["prefixItems"] = [annotation_to_json_schema(a) for a in type_args]
            schema["minItems"] = len(type_args)
            schema["maxItems"] = len(type_args)
        return schema

    if origin in (dict, dict):
        type_args = get_args(annotation)
        schema = {"type": "object"}
        if len(type_args) == 2 and type_args[0] is not str:
            # 非字符串键无法直接用 JSON Schema 表达，退回宽松的 object
            return {"type": "object"}
        if len(type_args) == 2:
            schema["additionalProperties"] = annotation_to_json_schema(type_args[1])
        return schema

    # --- 常见标准库标量 ---------------------------------------------------
    for py_type, fmt in ((_dt.datetime, "date-time"),
                         (_dt.date, "date"),
                         (_uuid.UUID, "uuid")):
        if annotation is py_type:
            return {"type": "string", "format": fmt}

    # 交给 pydantic 兜底（能处理 NewType、Annotated 等）
    try:
        from pydantic import TypeAdapter

        return _strip_pydantic_internals(TypeAdapter(annotation).json_schema())
    except Exception:
        return {}


def _json_type_name(py_type: type) -> str:
    if py_type is bool:
        return "boolean"
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is str:
        return "string"
    if py_type in (list, tuple, set):
        return "array"
    if py_type is dict:
        return "object"
    return "string"


def _strip_pydantic_internals(schema: dict[str, Any]) -> dict[str, Any]:
    """去掉 pydantic 自动加的 title/$defs 噪音，让 schema 更短、更省 token。"""
    schema.pop("title", None)
    schema.pop("$defs", None)
    for key in ("properties",):
        nested = schema.get(key)
        if isinstance(nested, dict):
            for value in nested.values():
                if isinstance(value, dict):
                    value.pop("title", None)
    for key in ("items", "additionalProperties"):
        nested = schema.get(key)
        if isinstance(nested, dict):
            nested.pop("title", None)
    return schema


# ---------------------------------------------------------------- docstring


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Parse a Google- or numpy-style docstring.

    Returns
    -------
    tuple[str, dict[str, str]]
        ``(summary, {param_name: param_description})``.
    """
    if not doc:
        return "", {}

    lines = doc.expandtabs(4).splitlines()
    summary_lines: list[str] = []
    param_docs: dict[str, str] = {}
    i = 0

    # 1) 收集摘要：第一个空行分隔标题段之前的连续非空行
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            if summary_lines:
                i += 1
                break
            i += 1
            continue
        if _is_section_header(lines, i):
            break
        summary_lines.append(line)
        i += 1

    summary = " ".join(summary_lines).strip()

    # 2) 解析参数段
    while i < len(lines):
        header = lines[i].strip()
        if _is_numpy_header(lines, i):
            i = _parse_numpy_params(lines, i + 2, param_docs)
            continue
        if header.lower().rstrip(":") in _SECTION_ALIASES:
            i = _parse_google_params(lines, i + 1, param_docs)
            continue
        i += 1

    return summary, param_docs


def _is_section_header(lines: list[str], index: int) -> bool:
    """True for any docstring section header we must not mistake for a param."""
    header = lines[index].strip().lower().rstrip(":")
    return header in _SECTION_ALIASES or header in _OTHER_SECTIONS


def _is_numpy_header(lines: list[str], index: int) -> bool:
    """True for a numpy-style ``Section\\n--------`` header."""
    title = lines[index].strip()
    if not title.replace(" ", "").isalpha():
        return False
    nxt = lines[index + 1].strip() if index + 1 < len(lines) else ""
    return bool(nxt) and set(nxt) == {"-"}


def _indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def _parse_google_params(lines: list[str], start: int, out: dict[str, str]) -> int:
    """Parse an indented ``name: description`` block. Returns next index."""
    i = start
    current: str | None = None
    block_indent: int | None = None

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()

        if stripped:
            indent = _indent_of(raw)
            if block_indent is None:
                block_indent = indent

            # 遇到下一个 section 就收工，避免把 "Returns:" 当成参数
            if _is_section_header(lines, i):
                break
            if indent < block_indent:
                break

            match = _GOOGLE_PARAM_RE.match(stripped)
            # 只有与首个参数同缩进的行才算新参数，更深的一律视为续行
            if match and indent == block_indent:
                current = match.group("name").lstrip("*")
                out[current] = match.group("desc").strip()
                i += 1
                continue

            if current is not None:
                out[current] = f"{out[current]} {stripped}".strip()
        i += 1

    return i


def _parse_numpy_params(lines: list[str], start: int, out: dict[str, str]) -> int:
    """Parse ``name : type`` lines whose description sits on deeper lines."""
    i = start
    current: str | None = None
    block_indent: int | None = None

    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if not stripped:
            i += 1
            continue

        indent = _indent_of(raw)
        if _is_numpy_header(lines, i) or _is_section_header(lines, i):
            break
        if block_indent is None:
            block_indent = indent

        match = _NUMPY_PARAM_RE.match(stripped)
        if match and indent == block_indent:
            current = match.group("name").lstrip("*")
            out[current] = ""
            i += 1
            continue

        if current is not None and indent > block_indent:
            out[current] = f"{out[current]} {stripped}".strip()
            i += 1
            continue

        break  # 缩进回到段落之外，说明参数段结束

    return i


# ---------------------------------------------------------------- 参数 schema


def build_parameters_schema(func: Any) -> dict[str, Any]:
    """Build the JSON Schema ``parameters`` object for a callable."""
    if not callable(func):
        raise SchemaError(f"{func!r} is not callable")

    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Cannot inspect signature of {func!r}: {exc}") from exc

    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    _, param_docs = parse_docstring(inspect.getdoc(func))

    properties: dict[str, Any] = {}
    required: list[str] = []

    for index, (name, param) in enumerate(signature.parameters.items()):
        if index == 0 and name in _SELF_LIKE:
            continue  # 类方法：跳过 self / cls
        if param.kind in (inspect.Parameter.VAR_POSITIONAL,
                          inspect.Parameter.VAR_KEYWORD):
            continue

        annotation = hints.get(name, param.annotation)
        param_schema = annotation_to_json_schema(annotation)

        if param_docs.get(name):
            param_schema["description"] = param_docs[name]

        if param.default is not inspect.Parameter.empty:
            param_schema["default"] = _jsonable_default(param.default)
        else:
            required.append(name)

        properties[name] = param_schema

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required
    # 默认不允许模型编造未声明的参数
    parameters["additionalProperties"] = False
    return parameters


def _jsonable_default(value: Any) -> Any:
    """Make a default value safe to embed into JSON Schema."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable_default(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable_default(v) for k, v in value.items()}
    return None
