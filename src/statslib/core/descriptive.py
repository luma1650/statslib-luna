from collections import Counter
from collections.abc import Iterable
from typing import Any, TypeVar, Protocol


class Comparable(Protocol):
    def __lt__(self, other) -> bool:
        pass

    def __eq__(self, other) -> bool:
        pass

    def __le__(self, other) -> bool:
        pass


C = TypeVar("C", bound=Comparable)


def _iter_len(iterable: Iterable[float]) -> int:
    n = 0
    for _ in iterable:
        n += 1

    return n


def mean(iterable: Iterable[float]) -> float:
    """Return the arithmetic mean of an iterable."""
    n = 0
    cur_sum = 0
    for item in iterable:
        cur_sum += item
        n += 1

    if n == 0:
        raise ValueError("mean requires at least one data point")

    return cur_sum / n


def median(iterable: Iterable[float]) -> float:
    data = list(iterable)
    n = len(data)

    if n == 0:
        raise ValueError("median requires at least one data point")

    if n % 2 == 1:
        return quickselect(data, n // 2)

    val1 = quickselect(data, n // 2 - 1)
    val2 = quickselect(data, n // 2)

    return (val1 + val2) / 2


def mode[T](iterable: Iterable[T]) -> T:
    """Return the first mode of an iterable."""
    data = list(iterable)

    if not data:
        raise ValueError("Mode of empty iterable")

    # O(n) expected: hashable objects
    try:
        counter = Counter(data)
    except TypeError:
        pass
    else:
        return counter.most_common(1)[0][0]

    # O(n log n): orderable objects
    try:
        ordered = sorted(data)
    except TypeError:
        pass
    else:
        best_value = ordered[0]
        best_count = 1
        current_count = 1

        for i in range(1, len(ordered)):
            if ordered[i] == ordered[i - 1]:
                current_count += 1
            else:
                current_count = 1

            if current_count > best_count:
                best_count = current_count
                best_value = ordered[i]

        return best_value

    # O(n²): equality-only objects
    best_value = data[0]
    best_count = 0

    for value in data:
        count = sum(item == value for item in data)

        if count > best_count:
            best_count = count
            best_value = value

    return best_value


def multimode[T](iterable: Iterable[T]) -> list[T]:
    data = list(iterable)

    if not data:
        return data

    # O(n) expected: hashable objects
    try:
        counter = Counter(data)
    except TypeError:
        pass
    else:
        target_count = counter.most_common(1)[0][1]
        return [value for value in counter if counter[value] == target_count]

    # O(n log n): orderable objects
    modes = []
    try:
        ordered = sorted(data)
    except TypeError:
        pass
    else:
        best_value = ordered[0]
        best_count = 1
        current_count = 1

        for i in range(1, len(ordered)):
            if ordered[i] == ordered[i - 1]:
                current_count += 1
            else:
                current_count = 1

            if current_count > best_count:
                best_count = current_count
                modes = [ordered[i]]
            elif current_count == best_count:
                modes.append(ordered[i])

        return modes

    # O(n²): equality-only objects
    best_count = 0

    for value in data:
        count = sum(item == value for item in data)

        if count > best_count:
            best_count = count
            modes = [value]
        elif count == best_count:
            modes.append(value)

    return modes


def weighted_mean(iterable: Iterable[float], weights : Iterable[float]) -> float:
    total_weight = 0.0
    sum_weight = 0.0
    for item, weight in zip(iterable, weights, strict=True):
        sum_weight += item * weight
        total_weight += weight

    try:
        return sum_weight / total_weight
    except ZeroDivisionError:
        raise ValueError("Invalid weights")



def quickselect(data: list[C], k: int) -> C:
    if not 0 <= k < len(data):
        raise IndexError("k is out of range")
    elif k == 0:
        return min(data)
    elif k == len(data) - 1:
        return max(data)

    left = 0
    right = len(data) - 1

    while left <= right:
        pivot_index = partition(data, left, right)

        if pivot_index == k:
            return data[k]
        elif pivot_index > k:
            right = pivot_index - 1
        else:
            left = pivot_index + 1

    raise RuntimeError("Quickselect failed")


def partition(data: list[C], left: int, right: int) -> int:
    """Partition a list into two halves, for Quicksort/Quickselect. Return the index of the pivot."""
    pivot = data[right]
    i = left
    for j in range(left, right):
        if data[j] <= pivot:
            data[i], data[j] = data[j], data[i]
            i += 1

    data[i], data[right] = data[right], data[i]
    return i