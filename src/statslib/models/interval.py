from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class Interval:
    """Immutable numeric interval."""

    lower: float
    upper: float
    closed_left: bool = True
    closed_right: bool = False

    def __post_init__(self) -> None:
        if not isfinite(self.lower) or not isfinite(self.upper):
            raise ValueError("Interval bounds must be finite")

        if self.lower >= self.upper:
            raise ValueError(
                "Interval lower bound must be less than upper bound"
            )

    @property
    def width(self) -> float:
        """Return the interval width."""
        return self.upper - self.lower

    @property
    def midpoint(self) -> float:
        """Return the interval midpoint (class mark)."""
        return self.lower / 2 + self.upper / 2

    def contains(self, value: float) -> bool:
        """Check whether a value belongs to the interval."""
        left = (
            value >= self.lower
            if self.closed_left
            else value > self.lower
        )

        right = (
            value <= self.upper
            if self.closed_right
            else value < self.upper
        )

        return left and right

    def __contains__(self, value: float) -> bool:
        return self.contains(value)

    def __str__(self) -> str:
        left = "[" if self.closed_left else "("
        right = "]" if self.closed_right else ")"

        return f"{left}{self.lower}, {self.upper}{right}"

    @classmethod
    def closed(cls, lower: float, upper: float) -> "Interval":
        """Create a closed interval [a, b]."""
        return cls(lower, upper, True, True)

    @classmethod
    def open(cls, lower: float, upper: float) -> "Interval":
        """Create an open interval (a, b)."""
        return cls(lower, upper, False, False)

    @classmethod
    def left_closed(cls, lower: float, upper: float) -> "Interval":
        """Create a left-closed interval [a, b)."""
        return cls(lower, upper, True, False)

    @classmethod
    def right_closed(cls, lower: float, upper: float) -> "Interval":
        """Create a right-closed interval (a, b]."""
        return cls(lower, upper, False, True)

    def __hash__(self) -> int:
        return hash((self.lower, self.upper, self.closed_left, self.closed_right))
