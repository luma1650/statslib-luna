
from collections.abc import Iterable, Iterator, Sequence
from math import sqrt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rows import GroupedFrequencyRow
from statslib.models.interval import Interval
from statslib.abc.frequency import AbstractFrequencyTable


class GroupedFrequencyTable(AbstractFrequencyTable[Interval]):
    """Immutable frequency table for grouped numerical data."""

    def __init__(
        self,
        data: Iterable[float],
        intervals: Sequence[Interval],
    ) -> None:
        values = tuple(data)
        bins = tuple(intervals)

        if not bins:
            raise ValueError("At least one interval is required")

        if any(
            a.upper > b.lower
            or (
                a.upper == b.lower
                and a.closed_right
                and b.closed_left
            )
            for a, b in zip(bins, bins[1:])
        ):
            raise ValueError(
                "Intervals must be sorted and non-overlapping"
            )

        counts = {interval: 0 for interval in bins}

        if len(counts) != len(bins):
            raise ValueError("Duplicate intervals are not allowed")

        for value in values:
            matches = [interval for interval in bins if value in interval]

            if len(matches) != 1:
                raise ValueError(
                    f"Value {value!r} does not belong to exactly one interval"
                )

            counts[matches[0]] += 1

        self._intervals = bins
        self._counts = counts
        self._total = len(values)

    def __len__(self) -> int:
        """Return the number of intervals."""
        return len(self._intervals)

    def __iter__(self) -> Iterator[Interval]:
        return iter(self._intervals)

    def __getitem__(self, interval: Interval) -> int:
        return self._counts[interval]

    @property
    def total(self) -> int:
        """Return the number of observations."""
        return self._total

    @property
    def intervals(self) -> tuple[Interval, ...]:
        return self._intervals

    def cumulative_frequency(self, interval: Interval) -> int:
        total = 0

        for current in self._intervals:
            total += self[current]
            if current == interval:
                return total

        raise KeyError(interval)

    def density(self, interval: Interval) -> float:
        """Return absolute frequency density."""
        return self[interval] / interval.width

    @property
    def mean(self) -> float:
        """Return the approximate grouped mean."""
        if self.total == 0:
            raise ValueError("Empty frequency table")

        return sum(
            interval.midpoint * freq
            for interval, freq in self.items()
        ) / self.total

    def variance(self, ddof: int = 0) -> float:
        """Return the approximate grouped variance."""
        if isinstance(ddof, bool) or not isinstance(ddof, int):
            raise TypeError("ddof must be an integer")
        if ddof < 0:
            raise ValueError("ddof must be nonnegative")

        if self.total <= ddof:
            raise ValueError("Not enough observations")

        mean = self.mean

        return sum(
            freq * (interval.midpoint - mean) ** 2
            for interval, freq in self.items()
        ) / (self.total - ddof)

    def stdev(self, ddof: int = 0) -> float:
        return sqrt(self.variance(ddof))

    @property
    def modal_intervals(self) -> tuple[Interval, ...]:
        """Return all intervals with maximum frequency."""
        if self.total == 0:
            raise ValueError("Empty frequency table")

        maximum = max(self._counts.values())

        return tuple(
            interval for interval, freq in self.items()
            if freq == maximum
        )

    def __repr__(self) -> str:
        return (
            f"GroupedFrequencyTable("
            f"intervals={len(self)}, total={self.total})"
        )

    def rows(self) -> "tuple[GroupedFrequencyRow, ...]":
        """Return immutable GroupedFrequencyRow objects in interval order.

        Zero-count intervals are retained. Relative columns are zero when
        the sample is empty, allowing the interval structure to be shown.
        """
        from .rows import GroupedFrequencyRow
        cumulative = 0
        rows = []
        for interval, count in self.items():
            cumulative += count
            rows.append(GroupedFrequencyRow(interval, count, cumulative, self.total))
        return tuple(rows)

    def to_text(self, *, precision: int = 4) -> str:
        """Return an aligned grouped-frequency display table."""
        from .rows import GroupedFrequencyRow, render_text
        return render_text(self.rows(), GroupedFrequencyRow.headers, precision=precision)

    def _repr_html_(self) -> str:
        """Display grouped frequencies as a notebook HTML table."""
        from .rows import GroupedFrequencyRow, render_html
        return render_html(self.rows(), GroupedFrequencyRow.headers)
