"""Tests for JSON Schema generation from Python callables."""

from __future__ import annotations

import datetime
import enum
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

import pytest
from pydantic import BaseModel

from nexus_agents import build_parameters_schema, parse_docstring
from nexus_agents.schema import annotation_to_json_schema


class Colour(str, enum.Enum):
    RED = "red"
    BLUE = "blue"


class Point(BaseModel):
    x: int
    y: int


# ---------------------------------------------------------------- 注解 -> schema


@pytest.mark.parametrize(
    "annotation, expected",
    [
        (str, {"type": "string"}),
        (int, {"type": "integer"}),
        (float, {"type": "number"}),
        (bool, {"type": "boolean"}),
        (list[int], {"type": "array", "items": {"type": "integer"}}),
        (List[str], {"type": "array", "items": {"type": "string"}}),
        (dict[str, int], {"type": "object", "additionalProperties": {"type": "integer"}}),
        (Dict[str, Any], {"type": "object", "additionalProperties": {}}),
        (Set[int], {"type": "array", "uniqueItems": True, "items": {"type": "integer"}}),
        (tuple[int, ...], {"type": "array", "items": {"type": "integer"}}),
        (datetime.datetime, {"type": "string", "format": "date-time"}),
        (Colour, {"type": "string", "enum": ["red", "blue"]}),
        (Literal["a", "b"], {"type": "string", "enum": ["a", "b"]}),
        (Point, {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"],
        }),
    ],
)
def test_annotation_mapping(annotation, expected):
    assert annotation_to_json_schema(annotation) == expected


def test_optional_unwraps_to_inner_schema():
    assert annotation_to_json_schema(Optional[int]) == {"type": "integer"}
    # PEP 604 写法同样要能处理
    assert annotation_to_json_schema(int | None) == {"type": "integer"}


def test_union_produces_any_of():
    schema = annotation_to_json_schema(Union[int, str])
    assert schema == {"anyOf": [{"type": "integer"}, {"type": "string"}]}


def test_unknown_annotation_degrades_to_any():
    class Weird:
        pass

    # 不应抛异常；返回 {} 表示「任意值」
    assert isinstance(annotation_to_json_schema(Weird), dict)


def test_fixed_length_tuple_uses_prefix_items():
    schema = annotation_to_json_schema(Tuple[int, str])
    assert schema["type"] == "array"
    assert schema["prefixItems"] == [{"type": "integer"}, {"type": "string"}]
    assert schema["minItems"] == schema["maxItems"] == 2


# ---------------------------------------------------------------- docstring


def test_google_style_docstring():
    summary, params = parse_docstring(
        """Convert a value.

        Args:
            value: the amount
            unit: target unit

        Returns:
            A string.
        """
    )
    assert summary == "Convert a value."
    assert params == {"value": "the amount", "unit": "target unit"}
    # "Returns:" 不能被当成参数
    assert "Returns" not in params


def test_numpy_style_docstring():
    summary, params = parse_docstring(
        """Convert a value.

        Parameters
        ----------
        value : float
            the amount
        unit : str
            target unit
        """
    )
    assert summary == "Convert a value."
    assert params["value"] == "the amount"
    assert params["unit"] == "target unit"


def test_empty_docstring():
    assert parse_docstring(None) == ("", {})
    assert parse_docstring("") == ("", {})


def test_multiline_param_description_is_joined():
    _, params = parse_docstring(
        """Summary.

        Args:
            value: the amount,
                in millimetres
        """
    )
    assert "millimetres" in params["value"]


# ---------------------------------------------------------------- 整体 schema


def test_build_parameters_schema_marks_required_and_defaults():
    def fn(a: int, b: str = "x", *, c: float = 1.5) -> None:
        """Do a thing.

        Args:
            a: first
            b: second
            c: third
        """

    schema = build_parameters_schema(fn)
    assert schema["type"] == "object"
    assert schema["required"] == ["a"]
    assert schema["properties"]["a"] == {"type": "integer", "description": "first"}
    assert schema["properties"]["b"]["default"] == "x"
    assert schema["properties"]["c"]["default"] == 1.5
    assert schema["additionalProperties"] is False


def test_build_parameters_schema_includes_enum_and_literal():
    def fn(unit: Colour, mode: Literal["min", "max"] = "max") -> None:
        """Pick.

        Args:
            unit: colour
            mode: which one
        """

    schema = build_parameters_schema(fn)
    assert schema["properties"]["unit"]["enum"] == ["red", "blue"]
    assert schema["properties"]["mode"]["enum"] == ["min", "max"]
    assert schema["required"] == ["unit"]


def test_self_is_skipped_for_methods():
    def method(self, a: int) -> None:
        """Bound method.

        Args:
            a: value
        """

    schema = build_parameters_schema(method)
    assert "self" not in schema["properties"]
    assert schema["required"] == ["a"]


def test_varargs_are_excluded():
    def fn(a: int, *args: int, **kwargs: str) -> None:
        """Varargs.

        Args:
            a: value
        """

    schema = build_parameters_schema(fn)
    assert set(schema["properties"]) == {"a"}


def test_non_callable_raises():
    from nexus_agents.errors import SchemaError

    with pytest.raises(SchemaError):
        build_parameters_schema(42)


def test_optional_param_is_not_required():
    def fn(target: Optional[str] = None) -> None:
        """Describe.

        Args:
            target: thing
        """

    schema = build_parameters_schema(fn)
    assert "required" not in schema or schema.get("required") == []
    assert schema["properties"]["target"]["default"] is None
