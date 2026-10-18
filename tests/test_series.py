
import math
import random
import statistics

import pytest

from statslib.series import StatsSeries
from statslib.abc.series import AbstractStatsSeries
from statslib.frequency.table import FrequencyTable


# ==========================================
# ABSTRACT BASE CLASS
# ==========================================

def test_abstract_series_inheritance():
    series = StatsSeries([1, 2, 3])

    assert isinstance(series, AbstractStatsSeries)


def test_abstract_series_cannot_be_instantiated():
    with pytest.raises(TypeError):
        AbstractStatsSeries([1, 2, 3])


def test_abstract_series_slicing():
    series = StatsSeries([1, 2, 3, 4, 5])

    subset = series[1:4]

    assert isinstance(subset, StatsSeries)
    assert isinstance(subset, AbstractStatsSeries)
    assert list(subset) == [2, 3, 4]


# ==========================================
# DATA RANGE
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1], 0),
    ([1, 2, 3, 4, 5], 4),
    ([-10, -5, 0, 5], 15),
    ([5, 5, 5], 0),
    ([1.5, 2.5, 10.5], 9),
    ([100, -100], 200),
])
def test_data_range(data, expected):
    series = StatsSeries(data)

    assert series.data_range == pytest.approx(expected)


def test_data_range_empty():
    with pytest.raises(ValueError):
        _ = StatsSeries([]).data_range


# ==========================================
# VARIANCE
# ==========================================

@pytest.mark.parametrize("data, population, sample", [
    ([1, 2, 3], 2 / 3, 1),
    ([1, 2, 3, 4, 5], 2, 2.5),
    ([5, 5, 5], 0, 0),
    ([10, 20], 25, 50),
    ([-2, 0, 2], 8 / 3, 4),
])
def test_variance(data, population, sample):
    series = StatsSeries(data)

    assert series.variance(ddof=0) == pytest.approx(population)
    assert series.variance(ddof=1) == pytest.approx(sample)


def test_variance_single_observation():
    series = StatsSeries([42])

    assert series.variance(ddof=0) == 0

    with pytest.raises(ValueError):
        series.variance(ddof=1)


@pytest.mark.parametrize("ddof", [-1, 3, 4, 100])
def test_variance_invalid_ddof(ddof):
    series = StatsSeries([1, 2, 3])

    with pytest.raises(ValueError):
        series.variance(ddof=ddof)


@pytest.mark.parametrize("ddof", [0.5, "1", None, True])
def test_variance_invalid_ddof_type(ddof):
    series = StatsSeries([1, 2, 3])

    with pytest.raises(TypeError):
        series.variance(ddof=ddof)


# ==========================================
# STANDARD DEVIATION
# ==========================================

@pytest.mark.parametrize("data", [
    [1, 2],
    [1, 2, 3, 4, 5],
    [10, 10, 10],
    [-10, 0, 10],
    [1.5, 2.5, 4.5],
])
def test_stdev_against_stdlib(data):
    series = StatsSeries(data)

    assert series.stdev(ddof=0) == pytest.approx(
        statistics.pstdev(data)
    )

    assert series.stdev(ddof=1) == pytest.approx(
        statistics.stdev(data)
    )


def test_stdev_is_square_root_of_variance():
    series = StatsSeries([2, 4, 8, 16])

    for ddof in (0, 1):
        assert series.stdev(ddof) == pytest.approx(
            math.sqrt(series.variance(ddof))
        )


# ==========================================
# MEAN ABSOLUTE DEVIATION
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1], 0),
    ([1, 2, 3], 2 / 3),
    ([1, 2, 3, 4, 5], 1.2),
    ([5, 5, 5], 0),
    ([0, 10], 5),
    ([-2, 0, 2], 4 / 3),
])
def test_mean_absolute_deviation(data, expected):
    series = StatsSeries(data)

    assert series.mean_absolute_deviation == pytest.approx(
        expected
    )


def test_mean_absolute_deviation_empty():
    with pytest.raises(ValueError):
        _ = StatsSeries([]).mean_absolute_deviation


# ==========================================
# MEDIAN ABSOLUTE DEVIATION
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1], 0),
    ([1, 2, 3], 1),
    ([1, 2, 3, 4, 5], 1),
    ([5, 5, 5], 0),
    ([0, 10], 5),
    ([1, 1, 2, 2, 4, 6, 9], 1),
])
def test_median_absolute_deviation(data, expected):
    series = StatsSeries(data)

    assert series.median_absolute_deviation == pytest.approx(
        expected
    )


def test_median_absolute_deviation_empty():
    with pytest.raises(ValueError):
        _ = StatsSeries([]).median_absolute_deviation


# ==========================================
# CACHE
# ==========================================

@pytest.mark.parametrize("attribute", [
    "mean",
    "median",
    "mode",
    "multimode",
    "data_range",
    "mean_absolute_deviation",
    "median_absolute_deviation",
    "frequency_table",
])
def test_cached_statistical_property(attribute):
    series = StatsSeries([1, 2, 2, 3, 4])

    first = getattr(series, attribute)
    second = getattr(series, attribute)

    assert first is second
    assert attribute in vars(series)


# ==========================================
# FREQUENCY TABLE INTEGRATION
# ==========================================

def test_frequency_table_integration():
    series = StatsSeries([2, 2, 3, 3, 3, 5])

    table = series.frequency_table

    assert isinstance(table, FrequencyTable)
    assert dict(table) == {2: 2, 3: 3, 5: 1}
    assert table.total == len(series)


def test_frequency_table_statistics_consistency():
    series = StatsSeries([2, 2, 3, 3, 3, 5])
    table = series.frequency_table

    assert table.mean == pytest.approx(series.mean)
    assert table.median == pytest.approx(series.median)
    assert table.mode == series.mode
    assert set(table.multimode) == set(series.multimode)


# ==========================================
# RANDOMIZED REFERENCE TESTS
# ==========================================

def test_randomized_dispersion():
    rng = random.Random(42)

    for n in range(2, 101):
        for _ in range(20):
            data = [
                rng.randint(-100, 100)
                for _ in range(n)
            ]

            series = StatsSeries(data)

            assert series.variance(0) == pytest.approx(
                statistics.pvariance(data)
            )

            assert series.variance(1) == pytest.approx(
                statistics.variance(data)
            )

            assert series.stdev(0) == pytest.approx(
                statistics.pstdev(data)
            )

            assert series.stdev(1) == pytest.approx(
                statistics.stdev(data)
            )

            assert series.data_range == max(data) - min(data)


# ==========================================
# NUMERICAL STABILITY
# ==========================================

def test_variance_large_offset():
    data = [
        1_000_000_000 + x
        for x in [-2, -1, 0, 1, 2]
    ]

    series = StatsSeries(data)

    assert series.variance(0) == pytest.approx(2.0)
    assert series.variance(1) == pytest.approx(2.5)


def test_variance_does_not_mutate_series():
    series = StatsSeries([5, 1, 4, 2, 3])
    original = list(series)

    series.variance()
    series.stdev()
    _ = series.mean_absolute_deviation
    _ = series.median_absolute_deviation

    assert list(series) == original
