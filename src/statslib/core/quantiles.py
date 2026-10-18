from collections.abc import Iterable
from math import ceil, isfinite, floor

from protocols import SupportsLE
from statslib.core.descriptive import quickselect


def quantile[T: SupportsLE](
    iterable: Iterable[T],
    k: float
) -> T:
    """
    Return the kth quantile using nearest rank.

    k must be between 0 and 1, inclusive.
    """
    if not isfinite(k) or not 0 <= k <= 1:
        raise ValueError("k must be between 0 and 1")

    data = list(iterable)

    if not data:
        raise ValueError("Quantile of an empty iterable")

    pos = max(0, ceil(k * len(data)) - 1)

    return quickselect(data, pos)

def quantile_linear(
    iterable: Iterable[float],
    k: float
) -> float:
    """Return the kth quantile using linear interpolation."""
    if not isfinite(k) or not 0 <= k <= 1:
        raise ValueError("k must be between 0 and 1")

    data = list(iterable)
    n = len(data)

    if n == 0:
        raise ValueError("Quantile of an empty iterable")

    h = (n - 1) * k
    lo = floor(h)
    hi = ceil(h)

    lower: float = quickselect(data, lo)

    if lo == hi:
        return float(lower)

    upper: float = quickselect(data, hi)
    return float(lower + (upper - lower) * (h - lo))