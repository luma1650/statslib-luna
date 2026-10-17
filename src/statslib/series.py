from collections.abc import Iterable, Iterator, Mapping, Sequence
from functools import cached_property
from typing import overload, TYPE_CHECKING
from statslib.core.descriptive import (
    mean,
    median,
    mode,
    multimode,
    weighted_mean,
)
if TYPE_CHECKING:
    from statslib.frequency import FrequencyTable


class StatsSeries[T](Sequence[T]):
    """Store observations in a tuple-backed statistical sequence.

    Preserve input order and repetitions. Integer indexing returns an
    observation; slicing returns a new instance of the same class.
    Statistical properties are evaluated lazily and cached.
    """
    def __init__(self, data):
        """Materialize data into a tuple, consuming the iterable once."""
        self._data = tuple(data)

    def __len__(self) -> int:
        """Return the number of observations, including repetitions."""
        return len(self._data)

    @overload
    def __getitem__(self, index: int, /) -> T: ...

    @overload
    def __getitem__(self, index: slice, /) -> "StatsSeries[T]": ...

    def __getitem__(
        self, index: int | slice, /
    ) -> "T | StatsSeries[T]":
        """Return an observation by index or a new series for a slice.

        Negative indices and slice bounds follow tuple semantics. An
        out-of-range integer index raises IndexError.
        """
        if isinstance(index, slice):
            return type(self)(self._data[index])

        return self._data[index]

    def __iter__(self) -> Iterator[T]:
        """Iterate over observations in their stored order."""
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        """Iterate over observations in reverse stored order."""
        for value in reversed(self._data):
            yield value

    def __eq__(self, other: object) -> bool:
        """Compare stored observations with another StatsSeries in order.

        Return NotImplemented for other operand types.
        """
        if not isinstance(other, StatsSeries):
            return NotImplemented

        return self._data == other._data

    def __hash__(self) -> int:
        """Return the tuple hash; all observations must be hashable."""
        return hash(self._data)

    @classmethod
    def from_frequencies(
        cls,
        mapping: Mapping[T, int]
    ) -> "StatsSeries[T]":
        """Expand a frequency mapping into a new series.

        Args:
            mapping: Observations mapped to nonnegative integer counts.

        Returns:
            A series grouped in mapping iteration order, omitting zero counts.

        Raises:
            TypeError: A frequency is not an integer or is a boolean.
            ValueError: A frequency is negative.

        Notes:
            Memory use grows with the sum of all frequencies.
        """
        data: list[T] = []

        for value, frequency in mapping.items():
            if isinstance(frequency, bool) or not isinstance(frequency, int):
                raise TypeError("Frequencies must be integers")

            if frequency < 0:
                raise ValueError("Frequencies cannot be negative")

            data.extend([value] * frequency)

        return cls(data)


    def __repr__(self) -> str:
        """Return a representation containing the stored observations."""
        return f"StatsSeries({list(self._data)!r})"

    @cached_property
    def mean(self) -> float:
        """Return the arithmetic mean; raise ValueError for an empty series."""
        return mean(self)

    @cached_property
    def median(self) -> float:
        """Return the middle value or the average of the middle pair.

        Observations must support ordering and, for even lengths, arithmetic.
        Raise ValueError for an empty series.
        """
        return median(self)

    @cached_property
    def mode(self) -> T:
        """Return a most frequent observation using descriptive.mode().

        Tie handling depends on whether observations are hashable, orderable,
        or equality-only. Raise ValueError for an empty series.
        """
        return mode(self)

    @cached_property
    def multimode(self) -> list[T]:
        """Return the list produced by descriptive.multimode().

        Return an empty list for an empty series. Ordering and duplicate
        handling follow the underlying function and observation capabilities.
        """
        return multimode(self)

    def weighted_mean(self, weights: Iterable[float]) -> float:
        """Return the mean weighted by one weight per observation.

        Args:
            weights: Numeric weights consumed alongside the observations.
                Negative weights are accepted.

        Raises:
            ValueError: Lengths differ or the total weight is zero.
        """
        return weighted_mean(self, weights)



    @cached_property
    def frequency_table(self) -> 'FrequencyTable[T]':
        """Return a cached frequency table of the hashable observations."""
        from statslib.frequency import FrequencyTable
        return FrequencyTable(self)