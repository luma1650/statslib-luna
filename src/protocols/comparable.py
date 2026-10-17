
"""Protocols for ordering comparisons."""

from typing import Any, Protocol


class SupportsLT(Protocol):
    def __lt__(self, other: Any, /) -> bool: ...


class SupportsLE(Protocol):
    def __le__(self, other: Any, /) -> bool: ...


class SupportsGT(Protocol):
    def __gt__(self, other: Any, /) -> bool: ...


class SupportsGE(Protocol):
    def __ge__(self, other: Any, /) -> bool: ...


class Comparable(
    SupportsLT,
    SupportsLE,
    SupportsGT,
    SupportsGE,
    Protocol,
):
    """Supports all four ordering operators."""


__all__ = [
    "SupportsLT",
    "SupportsLE",
    "SupportsGT",
    "SupportsGE",
    "Comparable",
]
