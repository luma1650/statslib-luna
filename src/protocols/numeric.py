
"""Protocols for numeric operations."""

from typing import Any, Protocol, Self


class SupportsAdd(Protocol):
    def __add__(self, other: Any, /) -> Any: ...


class SupportsSub(Protocol):
    def __sub__(self, other: Any, /) -> Any: ...


class SupportsMul(Protocol):
    def __mul__(self, other: Any, /) -> Any: ...


class SupportsTrueDiv(Protocol):
    def __truediv__(self, other: Any, /) -> Any: ...


class SupportsAbs(Protocol):
    def __abs__(self) -> Self: ...


class SupportsFloat(Protocol):
    def __float__(self) -> float: ...


class SupportsInt(Protocol):
    def __int__(self) -> int: ...


class SupportsRound(Protocol):
    def __round__(
        self, ndigits: int = 0, /
    ) -> Any: ...


__all__ = [
    "SupportsAdd",
    "SupportsSub",
    "SupportsMul",
    "SupportsTrueDiv",
    "SupportsAbs",
    "SupportsFloat",
    "SupportsInt",
    "SupportsRound",
]
