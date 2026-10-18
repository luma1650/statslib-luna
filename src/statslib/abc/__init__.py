"""Public abstract interfaces for statistical containers."""
from .frequency import AbstractFrequencyTable
from .series import AbstractStatsSeries

__all__ = ["AbstractFrequencyTable", "AbstractStatsSeries"]
