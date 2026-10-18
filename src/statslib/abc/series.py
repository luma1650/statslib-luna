
"""Abstract interfaces for statistical series."""

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import Self, TYPE_CHECKING, overload, Any

from statslib.core.descriptive import (
    mean,
    median,
    mode,
    multimode,
    weighted_mean,
)
from statslib.core.dispersion import (
    data_range,
    variance,
    stdev,
    mean_absolute_deviation,
    median_absolute_deviation,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from statslib.frequency.table import FrequencyTable


class AbstractStatsSeries[T](Sequence[T], ABC):
    """
    Abstract base class for statistical sequences.

    Provides shared statistical calculations while leaving
    storage, indexing and mutability to concrete subclasses.
    """

    @abstractmethod
    def __init__(self, data: Iterable[T]) -> None:
        """Initialize a series from observations."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of observations."""
        ...

    @overload
    def __getitem__(self, index: int, /) -> T: ...

    @overload
    def __getitem__(self, index: slice, /) -> Self: ...

    @abstractmethod
    def __getitem__(self, index: int | slice, /) -> T | Self:
        """Return an observation or a sliced series."""
        ...

    @classmethod
    def from_frequencies(
        cls,
        mapping: Mapping[T, int],
    ) -> Self:
        """Construct a series from absolute frequencies."""
        data: list[T] = []

        for value, frequency in mapping.items():
            if isinstance(frequency, bool) or not isinstance(frequency, int):
                raise TypeError("Frequencies must be integers")

            if frequency < 0:
                raise ValueError("Frequencies cannot be negative")

            data.extend([value] * frequency)

        return cls(data)

    # ======================================
    # CENTRAL TENDENCY
    # ======================================

    @property
    def mean(self) -> float:
        """Return the arithmetic mean."""
        return mean(self)

    @property
    def median(self) -> float:
        """Return the median."""
        return median(self)

    @property
    def mode(self) -> T:
        """Return a mode of the series."""
        return mode(self)

    @property
    def multimode(self) -> list[T]:
        """Return all modal observations."""
        return multimode(self)

    def weighted_mean(
        self,
        weights: Iterable[float],
    ) -> float:
        """Return the weighted arithmetic mean."""
        return weighted_mean(self, weights)

    # ======================================
    # DISPERSION
    # ======================================

    @property
    def data_range(self) -> float:
        """Return the difference between maximum and minimum."""
        return data_range(self)

    def variance(self, ddof: int = 0) -> float:
        """Return the variance with the specified ddof."""
        return variance(self, ddof)

    def stdev(self, ddof: int = 0) -> float:
        """Return the standard deviation."""
        return stdev(self, ddof)

    @property
    def mean_absolute_deviation(self) -> float:
        """Return the mean absolute deviation about the mean."""
        return mean_absolute_deviation(self)

    @property
    def median_absolute_deviation(self) -> float:
        """Return the median absolute deviation about the median."""
        return median_absolute_deviation(self)

    # ======================================
    # FREQUENCIES
    # ======================================

    @property
    def frequency_table(self) -> "FrequencyTable[T]":
        """Construct a frequency table from the observations."""
        from statslib.frequency.table import FrequencyTable
        return FrequencyTable(self)

    def quantile(self, k: float, *, method: str = "nearest") -> T | float:
        """Return a quantile using nearest rank or linear interpolation."""
        from statslib.core.quantiles import quantile, quantile_linear
        if method == "nearest":
            return quantile(self, k)
        if method == "linear":
            return quantile_linear(self, k)
        raise ValueError("method must be 'nearest' or 'linear'")

    def percentile(self, p: float, *, method: str = "nearest") -> T | float:
        """Return the percentile p, where 0 <= p <= 100."""
        return self.quantile(p / 100, method=method)

    @property
    def quartiles(self) -> tuple[float, float, float]:
        """Return Q1, Q2 and Q3 using linear interpolation."""
        return tuple(self.quantile(k, method="linear") for k in (.25, .5, .75))

    @property
    def interquartile_range(self) -> float:
        """Return Q3 minus Q1 using linear interpolation."""
        q1, _, q3 = self.quartiles
        return q3 - q1

    def coefficient_of_variation(self, ddof: int = 0) -> float:
        """Return standard deviation divided by absolute mean."""
        if self.mean == 0:
            raise ValueError("Coefficient of variation requires a nonzero mean")
        return self.stdev(ddof) / abs(self.mean)

    def origin_moment(self, n: int) -> float:
        """Return the nth raw empirical moment."""
        from statslib.core.moments import origin_moment
        return origin_moment(self, n)

    def central_moment(self, n: int) -> float:
        """Return the nth central empirical moment."""
        from statslib.core.moments import central_moment
        return central_moment(self, n)

    def standard_moment(self, n: int) -> float:
        """Return the nth standardized empirical moment."""
        from statslib.core.moments import standard_moment
        return standard_moment(self, n)

    @property
    def skewness(self) -> float:
        """Return unadjusted moment skewness; require positive variance."""
        return self.standard_moment(3)

    @property
    def excess_kurtosis(self) -> float:
        """Return unadjusted excess kurtosis; require positive variance."""
        return self.standard_moment(4) - 3

    def plot(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot observations against their zero-based index; return Axes.

        Matplotlib is optional. Pass ax to compose plots and kwargs to
        Axes.plot. Call pyplot.show() or ax.figure.savefig() yourself.
        """
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        ax.plot(range(len(self)), tuple(self), **kwargs)
        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        return ax

    def hist(self, bins=10, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot a numeric histogram; forward bins and kwargs to Axes.hist."""
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        ax.hist(tuple(self), bins=bins, **kwargs)
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
        ax.boxplot(tuple(self), **kwargs)
        return ax

    def ecdf(self, *, ax: "Axes | None" = None, **kwargs: Any) -> "Axes":
        """Plot the empirical CDF; forward kwargs to Axes.ecdf."""
        from statslib.plotting import get_axes, require_data
        require_data(self)
        ax = get_axes(ax)
        from statslib.plotting import draw_ecdf
        draw_ecdf(ax, tuple(self), **kwargs)
        return ax
