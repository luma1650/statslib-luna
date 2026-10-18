"""Storage-independent frequency-table contracts and shared operations."""
from abc import ABC
from collections import Counter
from collections.abc import Iterator, Mapping, ValuesView
from types import MappingProxyType


class AbstractFrequencyTable[T](Mapping[T, int], ABC):
    """Mapping observations to nonnegative integer counts.

    Implement __getitem__, __iter__ and __len__. Iteration/keys() yield
    observations; values()/frequencies() yield counts; items() yields pairs.
    __len__ counts keys (including zero-count grouped intervals), not sample
    size. Constructors and storage are deliberately left to subclasses.
    Defaults are not cached, so mutable implementations remain correct.
    """

    @property
    def total(self) -> int:
        """Return sample size, including repetitions."""
        return sum(self.values())

    def frequencies(self) -> ValuesView[int]:
        """Alias of Mapping.values(); observations are available via keys()."""
        return self.values()

    def iter_counts(self) -> Iterator[T]:
        """Expand counts lazily in encounter order, when explicitly requested."""
        for value, count in self.items():
            for _ in range(count):
                yield value

    @property
    def absolute(self) -> Mapping[T, int]:
        """Return an immutable snapshot of absolute frequencies."""
        return MappingProxyType(dict(self.items()))

    @property
    def relative(self) -> Mapping[T, float]:
        """Return normalized counts; zero-total tables have zero-valued entries."""
        total = self.total
        return MappingProxyType({value: count / total if total else 0.0
                                 for value, count in self.items()})

    def frequency(self, value: T) -> int:
        """Return the count, or zero for an absent observation."""
        return self.get(value, 0)

    def relative_frequency(self, value: T) -> float:
        """Return empirical probability mass; require nonempty observations."""
        if not self.total:
            raise ValueError("Relative frequency of an empty table")
        return self.frequency(value) / self.total

    def pmf(self, value: T) -> float:
        """Return empirical probability mass at value."""
        return self.relative_frequency(value)

    def percentage(self) -> Mapping[T, float]:
        """Return the complete percentage mapping, matching FrequencyTable."""
        return MappingProxyType({value: relative * 100
                                 for value, relative in self.relative.items()})

    def cumulative_frequency(self, value: T) -> int:
        """Count observations <= value; ordered observations are required.

        Grouped implementations override this to accumulate by interval
        order, since interval objects do not define numeric ordering.
        """
        return sum(count for key, count in self.items() if key <= value)

    def cdf(self, value: T) -> float:
        """Return cumulative probability, dispatching to the subclass's order."""
        if not self.total:
            raise ValueError("CDF of an empty table")
        return self.cumulative_frequency(value) / self.total

    def cumulative_relative_frequency(self, value: T) -> float:
        return self.cdf(value)

    @property
    def multimode(self) -> tuple[T, ...]:
        """Return tied positive-count modes in encounter order."""
        maximum = max(self.values(), default=0)
        if maximum == 0:
            return ()
        return tuple(value for value, count in self.items() if count == maximum)

    @property
    def mode(self) -> T:
        """Return the first mode; raise ValueError when no observations exist."""
        modes = self.multimode
        if not modes:
            raise ValueError("Mode of an empty table")
        return modes[0]

    def most_common(self, n: int | None = None) -> list[tuple[T, int]]:
        """Return up to n value-count pairs, preserving encounter order in ties."""
        if n is not None:
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError("n must be an integer or None")
            if n < 0:
                raise ValueError("n must be nonnegative")
        return Counter(self).most_common(n)
