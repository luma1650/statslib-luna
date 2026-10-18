from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import Self, overload, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from statslib.abc.series import AbstractStatsSeries
from statslib.algorithms.welford import WelfordVariance, WelfordMoments


class StatsSeries[T](AbstractStatsSeries[T]):
    """Immutable tuple-backed statistical series."""

    def __init__(self, data: Iterable[T]) -> None:
        self._data = tuple(data)

    def __len__(self) -> int:
        return len(self._data)

    @overload
    def __getitem__(self, index: int, /) -> T: ...

    @overload
    def __getitem__(self, index: slice, /) -> Self: ...


    def __getitem__(self, index: int | slice, /) -> T | Self:
        if isinstance(index, slice):
            return type(self)(self._data[index])
        return self._data[index]

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)

    def __reversed__(self) -> Iterator[T]:
        return reversed(self._data)

    def __repr__(self) -> str:
        return f"StatsSeries({list(self._data)!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StatsSeries):
            return NotImplemented
        return self._data == other._data

    def __hash__(self) -> int:
        return hash(self._data)

    # Cache only in immutable series.

    @cached_property
    def mean(self) -> float:
        return super().mean

    @cached_property
    def median(self) -> float:
        return super().median

    @cached_property
    def mode(self) -> T:
        return super().mode

    @cached_property
    def data_range(self) -> float:
        return super().data_range

    @cached_property
    def frequency_table(self):
        return super().frequency_table

    @cached_property
    def multimode(self) -> list[T]:
        return super().multimode

    @cached_property
    def mean_absolute_deviation(self) -> float:
        return super().mean_absolute_deviation

    @cached_property
    def median_absolute_deviation(self) -> float:
        return super().median_absolute_deviation

    @cached_property
    def _welford(self) -> WelfordVariance:
        accumulator = WelfordVariance()
        accumulator.extend(self)
        return accumulator

    def variance(self, ddof: int = 0) -> float:
        return self._welford.variance(ddof)

    def stdev(self, ddof: int = 0) -> float:
        return self._welford.std(ddof)
    @cached_property
    def quartiles(self) -> tuple[float, float, float]:
        # Sort once for all three quartiles and subsequent IQR queries.
        from math import floor, ceil
        ordered = sorted(self)
        if not ordered:
            raise ValueError("Quartiles of empty series")
        result = []
        for k in (.25, .5, .75):
            h = (len(ordered) - 1) * k
            lo, hi = floor(h), ceil(h)
            result.append(float(ordered[lo] + (ordered[hi] - ordered[lo]) * (h - lo)))
        return tuple(result)

    @cached_property
    def interquartile_range(self) -> float:
        return super().interquartile_range

    @cached_property
    def _moments(self) -> WelfordMoments:
        accumulator = WelfordMoments()
        accumulator.extend(self)
        return accumulator

    @cached_property
    def skewness(self) -> float:
        return self._moments.skewness

    @cached_property
    def excess_kurtosis(self) -> float:
        return self._moments.excess_kurtosis

    def plot(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot observations against their zero-based index; return Axes.

        Matplotlib is optional. Pass ax to compose plots and kwargs to
        Axes.plot. Call pyplot.show() or ax.figure.savefig() yourself.
        """
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        ax.plot(range(len(self)), self._data, **kwargs)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        return ax

    def hist(self, bins=10, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot a numeric histogram; forward bins and kwargs to Axes.hist."""
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        ax.hist(self._data, bins=bins, **kwargs)
        value_label = "Value"
        count_label = "Density" if kwargs.get("density", False) else "Frequency"
        if kwargs.get("orientation", "vertical") == "horizontal":
            ax.set_xlabel(count_label)
            ax.set_ylabel(value_label)
        else:
            ax.set_xlabel(value_label)
            ax.set_ylabel(count_label)
        return ax

    def boxplot(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot a box-and-whisker summary; forward kwargs to Axes.boxplot."""
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        ax.boxplot(self._data, **kwargs)
        return ax

    def ecdf(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot the empirical CDF; forward kwargs to Axes.ecdf."""
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        from statslib.plotting import draw_ecdf
        draw_ecdf(ax, self._data, **kwargs)
        return ax
