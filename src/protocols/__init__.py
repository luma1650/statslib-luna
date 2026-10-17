
"""Public protocols for statslib-luna."""

from .numeric import (
    SupportsAdd,
    SupportsSub,
    SupportsMul,
    SupportsTrueDiv,
    SupportsAbs,
    SupportsFloat,
    SupportsInt,
    SupportsRound,
)

from .comparable import (
    SupportsLT,
    SupportsLE,
    SupportsGT,
    SupportsGE,
    Comparable,
)

from .statistical import (
    SupportsMean,
    SupportsVariance,
    SupportsMedian,
    SupportsMode,
    SupportsCDF,
    SupportsPPF,
    SupportsPMF,
    SupportsPDF,
    SupportsSampling,
)

__all__ = [
    "SupportsAdd",
    "SupportsSub",
    "SupportsMul",
    "SupportsTrueDiv",
    "SupportsAbs",
    "SupportsFloat",
    "SupportsInt",
    "SupportsRound",
    "SupportsLT",
    "SupportsLE",
    "SupportsGT",
    "SupportsGE",
    "Comparable",
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
