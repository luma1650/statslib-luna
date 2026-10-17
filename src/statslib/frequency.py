from bisect import bisect_right
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Generator
from functools import cached_property, cache
from typing import Callable, Any

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
        self._cum = None
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

        obj._cum = None

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
            The first nonempty result is cached. Subsequent calls reuse it
            even when key or reverse differs. Values or sorting keys must
            support ordering when the cumulative mapping is first computed.
        """
        if not self:
            return frozendict()
        if self._cum is not None:
            return self._cum

        sorted_val = sorted(self, key=key, reverse=reverse)
        cum_sum = self[sorted_val[0]]
        mapping = {sorted_val[0]: cum_sum}
        for i in sorted_val[1:]:
            cum_sum += self[i]
            mapping[i] = cum_sum
        self._cum = frozendict(mapping)
        return self._cum

    def cumulative_relative(self, key: Callable[[T], Any] | None = None, reverse: bool = False) -> frozendict[T, float]:
        """Return cumulative counts divided by the total observation count.

        Args:
            key: Optional sorting key passed to cumulative().
            reverse: Whether to request descending observation order.

        Returns:
            An immutable mapping, empty for an empty table. The ordering
            follows cumulative(), including its first-call cache behavior.
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

    def find_pos(self, k: int, reverse: bool = False) -> T:
        """Return the observation at zero-based rank k without expanding counts.

        Args:
            k: Rank in the ordered sample, from zero to total - 1.
            reverse: Whether to request descending observation order.

        Raises:
            IndexError: The rank is outside the sample.

        Notes:
            Ordering follows cumulative(), including its first-call cache.
        """
        if not 0 <= k < self.total:
            raise IndexError("Position out of range")

        cumulative = self.cumulative(reverse=reverse)

        values = list(cumulative.keys())
        frequencies = list(cumulative.values())

        index = bisect_right(frequencies, k)

        return values[index]

    @cached_property
    def median(self) -> float:
        """Return the middle observation or the average of the middle pair.

        Raise ValueError for an empty table. Ordering follows the cached
        cumulative mapping; averaging requires arithmetic observations.
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