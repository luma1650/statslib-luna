from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Generator
from functools import cached_property
from math import ceil, floor, fsum, isfinite, sqrt
from typing import Callable, Any, TYPE_CHECKING, Self

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from protocols import SupportsGT, SupportsLT
from statslib.core.descriptive import weighted_mean
from statslib.series import StatsSeries


try:
    from builtins import frozendict  # Python 3.15+
except ImportError:
    from frozendict import frozendict  # PyPI


class FrequencyTable[T](Mapping[T, int]):
    """Map distinct hashable observations to their absolute frequencies.

    Iteration preserves first-encounter order, and len() counts distinct
    values. Statistical properties are computed lazily and cached.
    """
    def __init__(self, data: Iterable[T]) -> None:
        """Count observations from data, consuming the iterable once.

        Args:
            data: Iterable of hashable observations.
        """
        self._cum = {}
        self._counts: frozendict[T, int] = frozendict(
            Counter(data)
        )

    @classmethod
    def from_frequencies(
            cls,
            mapping: Mapping[T, int]
    ) -> "FrequencyTable[T]":
        """Build a table from explicit observation counts.

        Args:
            mapping: Hashable observations mapped to nonnegative integer counts.

        Returns:
            A new table with zero-frequency entries omitted.

        Raises:
            TypeError: A frequency is not an integer or is a boolean.
            ValueError: A frequency is negative.
        """
        obj = cls.__new__(cls)

        counts = dict(mapping)

        for frequency in counts.values():
            if isinstance(frequency, bool) or not isinstance(frequency, int):
                raise TypeError("Frequencies must be integers")

            if frequency < 0:
                raise ValueError("Frequencies cannot be negative")

        obj._counts = frozendict({
            value: frequency
            for value, frequency in counts.items()
            if frequency > 0
        })

        obj._cum = {}

        return obj

    def __getitem__(self, key: T) -> int:
        """Return the absolute frequency of key; raise KeyError if absent."""
        return self._counts[key]

    def __iter__(self) -> Iterator[T]:
        """Iterate over distinct observations in insertion order."""
        return iter(self._counts)

    def iter_counts(self) -> Generator[T, None, None]:
        """Yield each observation as many times as its absolute frequency.

        Observations are grouped in insertion order; original sample order
        is not reconstructed.
        """
        for key, value in self._counts.items():
            for _ in range(value):
                yield key

    def __len__(self) -> int:
        """Return the number of distinct observations."""
        return len(self._counts)

    def __repr__(self) -> str:
        """Return a representation containing the absolute frequency mapping."""
        return f"FrequencyTable({dict(self._counts)!r})"

    def __eq__(self, other: object) -> bool:
        """Compare counts with another FrequencyTable, ignoring insertion order.

        Return NotImplemented for other operand types.
        """
        if not isinstance(other, FrequencyTable):
            return NotImplemented

        return self._counts == other._counts

    def __hash__(self) -> int:
        """Return a hash derived from the immutable frequency mapping."""
        return hash(self._counts)

    @cached_property
    def total(self) -> int:
        """Return the total number of observations, including repetitions."""
        return sum(self._counts.values())

    @property
    def absolute(self) -> Mapping[T, int]:
        """Return the immutable mapping of observations to absolute frequencies."""
        return self._counts

    @cached_property
    def relative(self) -> frozendict[T, float]:
        """Return immutable frequencies normalized by the observation total.

        An empty table produces an empty mapping.
        """
        if self.total == 0:
            return frozendict()

        return frozendict({
            value: frequency / self.total
            for value, frequency in self._counts.items()
        })

    def cumulative(self, key: Callable[[T], SupportsLT | SupportsGT] | None = None, reverse: bool = False) -> frozendict[T, int]:
        """Return cumulative absolute frequencies in sorted observation order.

        Args:
            key: Optional sorting key, as accepted by sorted().
            reverse: Whether to sort observations in descending order.

        Returns:
            An immutable cumulative mapping, or an empty mapping if empty.

        Notes:
            Default ascending and descending orders are cached separately.
            Custom keys are recomputed and need not be hashable. Values or
            sorting keys must support ordering.
        """
        if not self:
            return frozendict()
        if key is None and reverse in self._cum:
            return self._cum[reverse]

        sorted_val = sorted(self, key=key, reverse=reverse)
        cum_sum = self[sorted_val[0]]
        mapping = {sorted_val[0]: cum_sum}
        for i in sorted_val[1:]:
            cum_sum += self[i]
            mapping[i] = cum_sum
        result = frozendict(mapping)
        if key is None:
            self._cum[reverse] = result
        return result

    def cumulative_relative(self, key: Callable[[T], Any] | None = None, reverse: bool = False) -> frozendict[T, float]:
        """Return cumulative counts divided by the total observation count.

        Args:
            key: Optional sorting key passed to cumulative().
            reverse: Whether to request descending observation order.

        Returns:
            An immutable mapping, empty for an empty table. The ordering
            follows cumulative(), using the requested sorting order.
        """
        return frozendict({value: frequency / self.total for value, frequency in self.cumulative(key, reverse).items()})


    def percentage(self) -> frozendict[T, float]:
        """Return a new immutable mapping of relative frequencies times 100.

        An empty table produces an empty mapping. Separate calls compute
        separate results; callers should compare values rather than identity.
        """
        return frozendict({value: frequency * 100 for value, frequency in self.relative.items()})


    def to_series(self) -> StatsSeries[T]:
        """Expand counts into a StatsSeries grouped in insertion order."""
        return StatsSeries.from_frequencies(self)

    @cached_property
    def mean(self) -> float:
        """Return the arithmetic mean using absolute frequencies as weights.

        Observations must support weighted arithmetic. Raise ValueError
        when the table is empty.
        """
        return weighted_mean(self, self._counts.values())

    @cached_property
    def mode(self) -> T:
        """Return the most frequent observation.

        Ties select the first inserted observation. Raise ValueError
        when the table is empty.
        """
        return max(self._counts, key=self._counts.__getitem__)

    @cached_property
    def _rank_index(self):
        ascending = self.cumulative()
        descending = self.cumulative(reverse=True)
        return {
            False: (tuple(ascending), tuple(ascending.values())),
            True: (tuple(descending), tuple(descending.values())),
        }

    def find_pos(self, k: int, reverse: bool = False) -> T:
        """Return the observation at zero-based rank k without expanding counts.

        Args:
            k: Rank in the ordered sample, from zero to total - 1.
            reverse: Whether to request descending observation order.

        Raises:
            IndexError: The rank is outside the sample.

        Notes:
            Ordering follows the requested ascending or descending order.
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("Position must be an integer")
        if not 0 <= k < self.total:
            raise IndexError("Position out of range")

        values, frequencies = self._rank_index[reverse]

        index = bisect_right(frequencies, k)

        return values[index]

    @cached_property
    def median(self) -> float:
        """Return the middle observation or the average of the middle pair.

        Raise ValueError for an empty table. Uses ascending observation
        order; averaging requires arithmetic observations.
        """
        n = self.total

        if n == 0:
            raise ValueError("Median of empty frequency table")

        if n % 2 == 1:
            return self.find_pos(n // 2)

        left = self.find_pos(n // 2 - 1)
        right = self.find_pos(n // 2)

        return (left + right) / 2

    @cached_property
    def multimode(self) -> tuple[T, ...]:
        """Return all most frequent observations as a tuple in insertion order.

        Return an empty tuple when the table is empty.
        """
        if not self:
            return ()

        maximum = max(self.values())

        return tuple(
            value
            for value, count in self.items()
            if count == maximum
        )
    def bar(
        self, *, relative: bool = False, ax: "Axes | None" = None,
        **kwargs: Any,
    ) -> "Axes":
        """Plot categorical bars in encounter order; return Axes.

        Positions are distinct even for keys with identical string labels.
        relative=True normalizes heights to sum to one. Other kwargs go
        to Axes.bar. Observations are never expanded.
        """
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        positions = list(range(len(self)))
        counts = self.relative if relative else self.absolute
        ax.bar(positions, list(counts.values()), **kwargs)
        ax.set_xticks(positions, [str(value) for value in self])
        ax.set_xlabel("Value")
        ax.set_ylabel("Relative frequency" if relative else "Frequency")
        return ax

    def hist(self, bins=10, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot a weighted numeric histogram without expanding counts.

        bins accepts an integer or explicit bin edges. Automatic bin-name
        estimators are unsupported with weighted data in Matplotlib.
        Weights come from this table and cannot be overridden.
        """
        from statslib.plotting import get_axes, require_data
        require_data(self)
        if "weights" in kwargs:
            raise TypeError("Histogram weights are provided by the frequency table")
        if isinstance(bins, str):
            raise ValueError("Use an integer or explicit bin edges for weighted histograms")
        ax = get_axes(ax)
        ax.hist(list(self), bins=bins, weights=list(self.values()), **kwargs)
        value_label = "Value"
        count_label = "Density" if kwargs.get("density", False) else "Frequency"
        if kwargs.get("orientation", "vertical") == "horizontal":
            ax.set_xlabel(count_label)
            ax.set_ylabel(value_label)
        else:
            ax.set_xlabel(value_label)
            ax.set_ylabel(count_label)
        return ax

    def ecdf(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot the weighted empirical CDF without expanding counts.

        Other kwargs go to Axes.ecdf; table weights cannot be overridden.
        """
        from statslib.plotting import get_axes, require_data
        require_data(self)
        if "weights" in kwargs:
            raise TypeError("ECDF weights are provided by the frequency table")
        ax = get_axes(ax)
        from statslib.plotting import draw_ecdf
        draw_ecdf(ax, list(self), weights=list(self.values()), **kwargs)
        return ax

    def plot(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot absolute or relative frequency bars; see bar()."""
        return self.bar(ax=ax, **kwargs)

    def frequency(self, value: T) -> int:
        """Return the count of value, or zero if absent."""
        return self._counts.get(value, 0)

    def relative_frequency(self, value: T) -> float:
        """Return empirical probability; require a nonempty table."""
        if not self.total:
            raise ValueError("Relative frequency of empty table")
        return self.frequency(value) / self.total

    def pmf(self, value: T) -> float:
        """Return empirical probability mass at value (zero if absent)."""
        return self.relative_frequency(value)

    def cumulative_frequency(self, value: T) -> int:
        """Return the number of observations <= value in O(log m) after sorting."""
        values, counts = self._rank_index[False]
        index = bisect_right(values, value)
        return counts[index - 1] if index else 0

    def cdf(self, value: T) -> float:
        """Return empirical P(X <= value); require a nonempty table."""
        if not self.total:
            raise ValueError("CDF of empty table")
        return self.cumulative_frequency(value) / self.total

    def cumulative_relative_frequency(self, value: T) -> float:
        """Return the cumulative relative frequency at value; alias of cdf."""
        return self.cdf(value)

    @cached_property
    def data_range(self) -> float:
        """Return largest minus smallest observation; require numeric data."""
        if not self:
            raise ValueError("Range of empty table")
        return max(self) - min(self)

    @cached_property
    def _squared_deviations(self) -> float:
        # Center on an observation before averaging to avoid loss of small
        # differences when values have a large common offset.
        if not self.total:
            raise ValueError("Variance of empty table")
        origin = next(iter(self))
        center = fsum((x - origin) * count for x, count in self.items()) / self.total
        residual = fsum(((x - origin) - center) * count for x, count in self.items())
        squared = fsum(((x - origin) - center) ** 2 * count for x, count in self.items())
        return max(0.0, squared - residual * (residual / self.total))

    def variance(self, ddof: int = 0) -> float:
        """Return count-weighted variance, without expanding observations."""
        if isinstance(ddof, bool) or not isinstance(ddof, int):
            raise TypeError("ddof must be an integer")
        if ddof < 0 or self.total <= ddof:
            raise ValueError("Insufficient observations")
        return self._squared_deviations / (self.total - ddof)

    def stdev(self, ddof: int = 0) -> float:
        """Return the square root of variance with the requested ddof."""
        return sqrt(self.variance(ddof))

    def coefficient_of_variation(self, ddof: int = 0) -> float:
        """Return standard deviation divided by absolute mean."""
        deviation = self.stdev(ddof)
        if self.mean == 0:
            raise ValueError("Coefficient of variation requires a nonzero mean")
        return deviation / abs(self.mean)

    def quantile(self, k: float, *, method: str = "nearest") -> T | float:
        """Return a nearest-rank or linearly interpolated quantile, 0 <= k <= 1.

        Nearest rank supports ordered categories; linear requires numeric
        observations. Sorting is cached, and counts are never expanded.
        """
        if method not in ("nearest", "linear"):
            raise ValueError("method must be 'nearest' or 'linear'")
        if not isfinite(k) or not 0 <= k <= 1:
            raise ValueError("k must be between 0 and 1")
        if not self.total:
            raise ValueError("Quantile of empty table")
        if k == 0 or k == 1:
            endpoint = self.find_pos(0 if k == 0 else self.total - 1)
            return float(endpoint) if method == "linear" else endpoint
        if method == "nearest":
            return self.find_pos(max(0, ceil(k * self.total) - 1))
        h = (self.total - 1) * k
        lo, hi = floor(h), ceil(h)
        lower = self.find_pos(lo)
        if lo == hi:
            return float(lower)
        upper = self.find_pos(hi)
        return float(lower + (upper - lower) * (h - lo))

    def percentile(self, p: float, *, method: str = "nearest") -> T | float:
        """Return percentile p, where 0 <= p <= 100."""
        return self.quantile(p / 100, method=method)

    @cached_property
    def quartiles(self) -> tuple[float, float, float]:
        """Return Q1, Q2 and Q3 using linear interpolation."""
        return tuple(self.quantile(k, method="linear") for k in (.25, .5, .75))

    @cached_property
    def interquartile_range(self) -> float:
        """Return Q3 minus Q1 using linear interpolation."""
        q1, _, q3 = self.quartiles
        return q3 - q1

    @cached_property
    def mean_absolute_deviation(self) -> float:
        """Return count-weighted absolute deviation about the mean."""
        center = self.mean
        return fsum(abs(x - center) * count for x, count in self.items()) / self.total

    @cached_property
    def median_absolute_deviation(self) -> float:
        """Return the median of absolute deviations, without expanding counts."""
        center = self.median
        deviations = Counter()
        for value, count in self.items():
            deviations[abs(value - center)] += count
        return type(self).from_frequencies(deviations).median

    def most_common(self, n: int | None = None) -> list[tuple[T, int]]:
        """Return up to n value-count pairs, largest counts first.

        Ties preserve encounter order. None returns all pairs; n must be
        a nonnegative integer. Categorical observations are supported.
        """
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError("n must be an integer or None")
            if n < 0:
                raise ValueError("n must be nonnegative")
        return Counter(self._counts).most_common(n)

    def merge(self, other: Mapping[T, int]) -> Self:
        """Return a new table with added counts; leave both inputs unchanged.

        Existing keys keep encounter order; new keys follow other. Counts
        must be nonnegative integers, as in from_frequencies().
        """
        validated = type(self).from_frequencies(other)
        counts = dict(self._counts)
        for value, count in validated.items():
            counts[value] = counts.get(value, 0) + count
        return type(self).from_frequencies(counts)
