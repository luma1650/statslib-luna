"""Measures of statistical dispersion."""

import math
from collections.abc import Iterable

from statslib.core.descriptive import mean, median
from statslib.algorithms.welford import WelfordVariance


def data_range(data: Iterable[float]) -> float:
    """Return the difference between the largest and smallest values."""
    iterator = iter(data)

    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Range of empty iterable") from None

    minimum = maximum = first

    for value in iterator:
        if value < minimum:
            minimum = value
        if value > maximum:
            maximum = value

    return maximum - minimum


def variance(
    data: Iterable[float],
    ddof: int = 0
) -> float:
    """Return variance using Welford's online algorithm."""
    accumulator = WelfordVariance()
    accumulator.extend(data)
    return accumulator.variance(ddof)


def stdev(
    data: Iterable[float],
    ddof: int = 0
) -> float:
    """Return standard deviation."""
    return math.sqrt(variance(data, ddof))


def mean_absolute_deviation(
    data: Iterable[float]
) -> float:
    """Return the mean absolute deviation about the mean."""
    values = list(data)

    if not values:
        raise ValueError("MAD of empty iterable")

    mu = mean(values)

    return math.fsum(
        abs(x - mu) for x in values
    ) / len(values)


def median_absolute_deviation(
    data: Iterable[float],
    scale: float = 1.0
) -> float:
    """Return the median absolute deviation about the median."""
    values = list(data)

    if not values:
        raise ValueError("MAD of empty iterable")

    center = median(values)

    return median(
        abs(x - center) for x in values
    ) * scale


def coefficient_of_variation(
    data: Iterable[float],
    ddof: int = 0
) -> float:
    """Return standard deviation divided by absolute mean."""
    accumulator = WelfordVariance()
    accumulator.extend(data)

    mu = accumulator.mean

    if mu == 0:
        raise ValueError(
            "Coefficient of variation requires a nonzero mean"
        )

    return accumulator.std(ddof) / abs(mu)
