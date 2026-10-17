from collections.abc import Iterable, Iterator, Mapping


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
