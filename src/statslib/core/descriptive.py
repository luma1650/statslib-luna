from collections import Counter
from collections.abc import Iterable
from typing import Any, TypeVar, Protocol

from protocols import SupportsLE


def _iter_len(iterable: Iterable[float]) -> int:
    """Count elements by consuming iterable once."""
    n = 0
    for _ in iterable:
        n += 1

    return n


def mean(iterable: Iterable[float]) -> float:
    """Return the arithmetic mean of numeric observations.

    Args:
        iterable: Observations supporting addition and division.

    Raises:
        ValueError: The iterable is empty.

    Notes:
        Consume the iterable once without materializing it.
    """
    n = 0
    cur_sum = 0
    for item in iterable:
        cur_sum += item
        n += 1

    if n == 0:
        raise ValueError("mean requires at least one data point")

    return cur_sum / n


def median(iterable: Iterable[float]) -> float:
    """Return the middle value or the average of the middle pair.

    Args:
        iterable: Orderable numeric observations, materialized internally.

    Raises:
        ValueError: The iterable is empty.

    Notes:
        Selection modifies only an internal list, leaving the input
        collection unchanged.
    """
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
    """Return a most frequent observation.

    Args:
        iterable: Observations, materialized internally.

    Raises:
        ValueError: The iterable is empty.

    Notes:
        Hashable observations use counting and select the first encountered
        value in a tie. Unhashable orderable observations use sorting and
        select the first tied value in sorted order. The equality-only
        fallback selects the first encountered tied value.
    """
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
    """Return the modal candidates produced by counting or comparison.

    Args:
        iterable: Observations, materialized internally.

    Returns:
        A list, empty when the input is empty.

    Notes:
        Hashable values produce distinct modes in encounter order. The
        sorting fallback accumulates candidates when a running count
        matches or exceeds the current best; its result may omit the first
        sorted value when all counts are one. The equality-only fallback
        can include repeated occurrences of the same modal value.
    """
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
    """Return the sum of weighted observations divided by total weight.

    Args:
        iterable: Numeric observations consumed once.
        weights: Numeric weights consumed alongside the observations.
            Negative weights are accepted.

    Raises:
        ValueError: The iterables have different lengths or the total
            weight is zero, including when both iterables are empty.
    """
    total_weight = 0.0
    sum_weight = 0.0
    for item, weight in zip(iterable, weights, strict=True):
        sum_weight += item * weight
        total_weight += weight

    try:
        return sum_weight / total_weight
    except ZeroDivisionError:
        raise ValueError("Invalid weights")



def quickselect[T: SupportsLE](data: list[T], k: int) -> T:
    """Select the observation at zero-based sorted rank k.

    Args:
        data: Mutable list of mutually orderable observations.
        k: Rank from zero to len(data) - 1.

    Returns:
        The value that would occupy rank k after sorting.

    Raises:
        IndexError: k is outside the list, including for an empty list.
        RuntimeError: The selection loop fails to locate the requested rank.

    Notes:
        Interior-rank selection partitions data in place. Endpoint ranks
        use min() or max() without rearranging the list.
    """
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


def partition(data: list[SupportsLE], left: int, right: int) -> int:
    """Partition an inclusive list segment around its rightmost value.

    Args:
        data: Mutable list whose elements support ordering.
        left: Inclusive start index of the segment.
        right: Inclusive end index and initial pivot position.

    Returns:
        The final pivot index. Values before it in the segment are less
        than or equal to the pivot; values after it are greater.

    Notes:
        Mutate data in place. Callers must provide valid bounds satisfying
        0 <= left <= right < len(data); bounds are not validated explicitly.
    """
    pivot = data[right]
    i = left
    for j in range(left, right):
        if data[j] <= pivot:
            data[i], data[j] = data[j], data[i]
            i += 1

    data[i], data[right] = data[right], data[i]
    return i