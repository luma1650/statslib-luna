"""Immutable presentation rows for ordinary and grouped frequencies."""
from dataclasses import dataclass
from html import escape
from statslib.models.interval import Interval


class _FrequencyMetrics:
    __slots__ = ()

    def __post_init__(self) -> None:
        for name in ("frequency", "cumulative_frequency", "total"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 0 <= self.frequency <= self.cumulative_frequency <= self.total:
            raise ValueError("Require 0 <= frequency <= cumulative_frequency <= total")

    @property
    def relative_frequency(self) -> float:
        """Return frequency / total, or zero for an empty sample."""
        return self.frequency / self.total if self.total else 0.0

    @property
    def cumulative_relative_frequency(self) -> float:
        return self.cumulative_frequency / self.total if self.total else 0.0

    @property
    def percentage(self) -> float:
        return 100 * self.relative_frequency

    @property
    def cumulative_percentage(self) -> float:
        return 100 * self.cumulative_relative_frequency

    def to_dict(self) -> dict:
        """Return a fresh mapping of displayed fields, preserving raw values."""
        return dict(zip(self.headers, self.cells()))

    def __str__(self) -> str:
        return " | ".join(_format(value, 4) for value in self.cells())

    def _repr_html_(self) -> str:
        return render_html((self,), self.headers)


@dataclass(frozen=True, slots=True)
class FrequencyRow[T](_FrequencyMetrics):
    """One display row; cumulative counts follow its parent table's row order."""
    value: T
    frequency: int
    cumulative_frequency: int
    total: int

    headers = ("Value", "Frequency", "Relative", "Cumulative", "Cumulative relative", "Percent", "Cumulative percent")

    def cells(self) -> tuple:
        return (self.value, self.frequency, self.relative_frequency,
                self.cumulative_frequency, self.cumulative_relative_frequency,
                self.percentage, self.cumulative_percentage)


@dataclass(frozen=True, slots=True)
class GroupedFrequencyRow(_FrequencyMetrics):
    """One grouped display row, including zero-count intervals."""
    interval: Interval
    frequency: int
    cumulative_frequency: int
    total: int

    headers = ("Interval", "Midpoint", "Width", "Frequency", "Relative", "Cumulative", "Cumulative relative", "Percent", "Cumulative percent", "Density")

    def __post_init__(self) -> None:
        if not isinstance(self.interval, Interval):
            raise TypeError("interval must be an Interval")
        _FrequencyMetrics.__post_init__(self)

    @property
    def midpoint(self) -> float:
        return self.interval.midpoint

    @property
    def width(self) -> float:
        return self.interval.width

    @property
    def density(self) -> float:
        """Return absolute frequency per unit interval width."""
        return self.frequency / self.width

    def cells(self) -> tuple:
        return (self.interval, self.midpoint, self.width, self.frequency,
                self.relative_frequency, self.cumulative_frequency,
                self.cumulative_relative_frequency, self.percentage,
                self.cumulative_percentage, self.density)


def _format(value, precision: int) -> str:
    return f"{value:.{precision}f}" if isinstance(value, float) else str(value)


def render_text(rows, headers, *, precision: int = 4) -> str:
    """Render an aligned plain-text table; keep every row, including zeros."""
    if isinstance(precision, bool) or not isinstance(precision, int):
        raise TypeError("precision must be an integer")
    if precision < 0:
        raise ValueError("precision must be nonnegative")
    lines = [headers, *(tuple(_format(v, precision) for v in row.cells()) for row in rows)]
    widths = [max(len(line[i]) for line in lines) for i in range(len(headers))]
    def format_line(line):
        return " | ".join(value.ljust(width) for value, width in zip(line, widths))
    return "\n".join([format_line(lines[0]), "-+-".join("-" * w for w in widths),
                      *(format_line(line) for line in lines[1:])])


def render_html(rows, headers) -> str:
    """Escape observation text before embedding it into notebook HTML."""
    head = "".join(f"<th>{escape(name)}</th>" for name in headers)
    body = "".join("<tr>" + "".join(f"<td>{escape(_format(v, 4))}</td>" for v in row.cells()) + "</tr>" for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"
