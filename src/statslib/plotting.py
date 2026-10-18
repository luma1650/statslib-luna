"""Shared plotting helpers; Matplotlib is imported only when plotting."""
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def get_axes(ax: "Axes | None" = None) -> "Axes":
    """Reuse an axes or create one without displaying a window."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            'Plotting requires Matplotlib. Install: pip install "statslib-luna[plot]"'
        ) from exc
    if ax is None:
        _, ax = plt.subplots()
    return ax


def require_data(data) -> None:
    if not data:
        raise ValueError("Cannot plot empty data")


def draw_ecdf(ax: "Axes", data, *, weights=None, compress=False, **kwargs) -> None:
    """Aggregate repeated values before ECDF compression.

    Some Matplotlib releases keep the first cumulative weight of a run
    when compressing duplicates. Summing weights first preserves the full
    empirical mass, including the endpoint at one.
    """
    if compress:
        import numpy as np
        values, inverse = np.unique(data, return_inverse=True)
        weights = np.bincount(inverse, weights=weights)
        data = values
    ax.ecdf(data, weights=weights, **kwargs)
