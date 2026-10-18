from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Mapping
from typing import Generic, TypeVar

T = TypeVar("T")


class AbstractFrequencyTable[T](ABC):
    """Abstract base class for frequency tables."""

    @abstractmethod
    def __init__(self, data: Iterable[T]) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of distinct values."""
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """Iterate over distinct values."""
        ...

    @abstractmethod
    def __getitem__(self, value: T) -> int:
        """Return the absolute frequency of a value."""
        ...

    @property
    @abstractmethod
    def total(self) -> int:
        """Return the total number of observations."""
        ...

    @abstractmethod
    def items(self) -> Iterable[tuple[T, int]]:
        """Return value-frequency pairs."""
        ...

    @abstractmethod
    def values(self) -> Iterable[T]:
        """Return distinct observed values."""
        ...

    @abstractmethod
    def frequencies(self) -> Iterable[int]:
        """Return absolute frequencies."""
        ...

    def frequency(self, value: T) -> int:
        """Return the absolute frequency of a value."""
        return self[value]

    def relative_frequency(self, value: T) -> float:
        """Return the relative frequency of a value."""
        if self.total == 0:
            raise ValueError("Relative frequency of an empty table")

        return self[value] / self.total

    def percentage(self, value: T) -> float:
        """Return the percentage frequency of a value."""
        return 100 * self.relative_frequency(value)

    def cumulative_frequency(self, value: T) -> int:
        """Return the cumulative frequency up to a value."""
        return sum(
            freq for key, freq in self.items()
            if key <= value
        )

    def cumulative_relative_frequency(self, value: T) -> float:
        """Return the cumulative relative frequency."""
        if self.total == 0:
            raise ValueError("Cumulative frequency of an empty table")

        return self.cumulative_frequency(value) / self.total

    def mode(self) -> tuple[T, ...]:
        """Return all modes of the distribution."""
        pairs = tuple(self.items())

        if not pairs:
            raise ValueError("Mode of an empty table")

        maximum = max(freq for _, freq in pairs)

        return tuple(
            value for value, freq in pairs
            if freq == maximum
        )

    def __contains__(self, value: object) -> bool:
        return any(key == value for key in self)

    def __bool__(self) -> bool:
        return self.total > 0