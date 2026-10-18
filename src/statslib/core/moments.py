from collections.abc import Iterable
from math import fsum
from .descriptive import mean


def _validate_order(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int):
        raise TypeError("Moment order must be an integer")
    if n < 0:
        raise ValueError("Moment order must be nonnegative")


def origin_moment(
    iterable: Iterable[float],
    n: int
) -> float:
    """Return the nth empirical raw moment."""
    _validate_order(n)

    data = list(iterable)

    if not data:
        raise ValueError("Moment of empty iterable")

    if n == 0:
        return 1.0

    if n == 1:
        return mean(data)

    return fsum(x ** n for x in data) / len(data)


def central_moment(
    iterable: Iterable[float],
    n: int
) -> float:
    """Return the nth empirical central moment."""
    _validate_order(n)

    data = list(iterable)

    if not data:
        raise ValueError("Moment of empty iterable")

    if n == 0:
        return 1.0

    if n == 1:
        return 0.0

    data_mean = mean(data)

    return fsum(
        (x - data_mean) ** n
        for x in data
    ) / len(data)


def standard_moment(
    iterable: Iterable[float],
    n: int
) -> float:
    """Return the nth standardized empirical moment."""
    _validate_order(n)

    data = list(iterable)

    if not data:
        raise ValueError("Moment of empty iterable")

    variance = central_moment(data, 2)

    if variance == 0:
        raise ValueError(
            "Standardized moments require positive variance"
        )

    if n == 0:
        return 1.0

    if n == 1:
        return 0.0

    if n == 2:
        return 1.0

    return central_moment(data, n) / variance ** (n / 2)
