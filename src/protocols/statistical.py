
"""Protocols for statistical objects and distributions."""

from typing import Protocol


# ==========================================
# DESCRIPTIVE STATISTICS
# ==========================================

class SupportsMean(Protocol):
    @property
    def mean(self) -> float: ...


class SupportsVariance(Protocol):
    @property
    def variance(self) -> float: ...


class SupportsMedian[T](Protocol):
    @property
    def median(self) -> T: ...


class SupportsMode[T](Protocol):
    @property
    def mode(self) -> T: ...


# ==========================================
# PROBABILITY DISTRIBUTIONS
# ==========================================

class SupportsCDF(Protocol):
    def cdf(self, x: float) -> float: ...


class SupportsPPF(Protocol):
    def ppf(self, p: float) -> float: ...


class SupportsPMF(Protocol):
    def pmf(self, x: int) -> float: ...


class SupportsPDF(Protocol):
    def pdf(self, x: float) -> float: ...


class SupportsSampling[T](Protocol):
    def sample(self, size: int = 1) -> list[T]: ...


__all__ = [
    "SupportsMean",
    "SupportsVariance",
    "SupportsMedian",
    "SupportsMode",
    "SupportsCDF",
    "SupportsPPF",
    "SupportsPMF",
    "SupportsPDF",
    "SupportsSampling",
]
