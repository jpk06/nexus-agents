"""Shared fixtures for the nexus-agents test suite."""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

import pytest

from nexus_agents import ToolRegistry, ToolSpec, tool


class Unit(str, Enum):
    CM = "cm"
    INCH = "inch"


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: the first number
        b: the second number
    """
    return a + b


@tool
def convert(value: float, unit: Unit = Unit.CM) -> str:
    """Convert a length to the given unit.

    Args:
        value: length in millimetres
        unit: target unit
    """
    return f"{value} -> {unit.value}"


@tool
def repeat(text: str, times: int = 2, *, sep: str = ",") -> str:
    """Repeat text a number of times.

    Args:
        text: text to repeat
        times: how many copies
        sep: separator inserted between copies
    """
    return sep.join([text] * times)


@tool
def boom() -> str:
    """Always raises."""
    raise ValueError("kaboom")


@tool
def find(items: list[int], mode: Literal["min", "max"] = "max") -> int:
    """Pick the smallest or largest item.

    Args:
        items: candidate numbers
        mode: which extreme to pick
    """
    return min(items) if mode == "min" else max(items)


@tool
def describe(target: Optional[str] = None) -> str:
    """Describe something, or nothing.

    Args:
        target: optional thing to describe
    """
    return target or "nothing"


@pytest.fixture
def calc_tools() -> ToolRegistry:
    return ToolRegistry([add, convert, repeat, boom, find, describe])


@pytest.fixture
def all_specs() -> list[ToolSpec]:
    return [add, convert, repeat, boom, find, describe]
