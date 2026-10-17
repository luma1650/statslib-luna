from collections.abc import Iterable, Iterator, Mapping
from statslib.core.descriptive import (
    mean,
    median,
    mode,
    multimode,
    weighted_mean,
)


class StatsSeries[T]:
    def __init__(self, data: Iterable[T]) -> None:
        self._data = tuple(data)

    @classmethod
    def from_frequencies(
            cls,
            mapping: Mapping[T, int]
    ) -> 'StatsSeries[T]':
        data: list[T] = []

        for value, frequency in mapping.items():
            if not isinstance(frequency, int) or isinstance(frequency, bool):
                raise TypeError("Frequencies must be integers")

            if frequency < 0:
                raise ValueError("Frequencies cannot be negative")

            data.extend([value] * frequency)

        return cls(data)


    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __repr__(self):
        return f"StatsSeries({list(self._data)!r})"

    @property
    def mean(self) -> float:
        return mean(self)

    @property
    def median(self) -> T:
        return median(self)

    @property
    def mode(self) -> T:
        return mode(self)

    @property
    def weighted_mean(self, weights: Iterable[float]) -> float:
        return weighted_mean(self, weights)

    @property
    def multimode(self) -> list[T]:
        return multimode(self)
