import math
from collections.abc import Iterable


class WelfordMoments:
    """
    Incrementally compute the first four central moments.

    Supports streaming observations in O(1) time per
    update and O(1) auxiliary memory.

    M2, M3 and M4 store unnormalized central moments.
    """

    def __init__(self, form: bool = True) -> None:
        self.n = 0
        self.M1 = 0.0
        self.M2 = 0.0
        self.M3 = 0.0
        self.M4 = 0.0
        self.form = form

    def push(self, x: float) -> None:
        """Update the moments with one observation."""
        self.n += 1
        n = self.n

        delta = x - self.M1
        delta_n = delta / n
        term1 = delta * delta_n * (n - 1)

        if self.form:
            delta_n2 = delta_n * delta_n
            self.M4 += (
                term1 * delta_n2 * (n*n - 3*n + 3)
                + 6 * delta_n2 * self.M2
                - 4 * delta_n * self.M3
            )

            self.M3 += (
                term1 * delta_n * (n - 2)
                - 3 * delta_n * self.M2
            )

        self.M2 += term1
        self.M1 += delta_n

    def extend(self, data: Iterable[float]) -> None:
        """Update the moments with multiple observations."""
        for x in data:
            self.push(x)

    @property
    def mean(self) -> float:
        if self.n == 0:
            raise ValueError("Mean requires at least one observation")
        return self.M1

    def variance(self, ddof: int = 1) -> float:
        """Return variance using the requested correction."""
        if not isinstance(ddof, int) or isinstance(ddof, bool):
            raise TypeError("ddof must be an integer")
        if ddof < 0 or self.n <= ddof:
            raise ValueError("Invalid degrees of freedom")

        return self.M2 / (self.n - ddof)

    def std(self, ddof: int = 1) -> float:
        """Return the standard deviation."""
        return math.sqrt(max(0.0, self.variance(ddof)))

    @property
    def skewness(self) -> float:
        """Return unadjusted moment skewness."""
        if not self.form:
            raise ValueError("Skewness required form calculation")
        if self.n == 0 or self.M2 <= 0:
            raise ValueError("Skewness requires positive variance")

        return (
            math.sqrt(self.n) * self.M3
            / self.M2 ** 1.5
        )

    @property
    def excess_kurtosis(self) -> float:
        """Return unadjusted excess kurtosis."""
        if not self.form:
            raise ValueError("Kurtosis required form calculation")
        if self.n == 0 or self.M2 <= 0:
            raise ValueError("Kurtosis requires positive variance")

        return (
            self.n * self.M4 / self.M2**2
        ) - 3


class WelfordVariance:
    """Incremental mean and variance accumulator."""

    def __init__(self) -> None:
        self.n = 0
        self.mean_value = 0.0
        self.M2 = 0.0

    def push(self, x: float) -> None:
        self.n += 1

        delta = x - self.mean_value
        self.mean_value += delta / self.n
        delta2 = x - self.mean_value

        self.M2 += delta * delta2

    def extend(self, data: Iterable[float]) -> None:
        for x in data:
            self.push(x)

    @property
    def mean(self) -> float:
        if self.n == 0:
            raise ValueError("Mean of empty data")
        return self.mean_value

    def variance(self, ddof: int = 0) -> float:
        if isinstance(ddof, bool) or not isinstance(ddof, int):
            raise TypeError("ddof must be an integer")

        if ddof < 0 or self.n <= ddof:
            raise ValueError("Insufficient observations")

        return self.M2 / (self.n - ddof)

    def std(self, ddof: int = 0) -> float:
        return math.sqrt(max(0.0, self.variance(ddof)))
