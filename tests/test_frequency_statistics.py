"""Reference and regression coverage for compressed table operations."""
import math
import random
import statistics
from unittest.mock import patch

import pytest

from statslib.frequency import FrequencyTable
from statslib import StatsSeries


def test_randomized_statistics_match_expanded_samples():
    rng = random.Random(72)
    for n in range(2, 90):
        data = [rng.randint(-25, 25) for _ in range(n)]
        table = FrequencyTable(data)
        series = StatsSeries(data)
        for ddof in (0, 1):
            expected = statistics.pvariance(data) if ddof == 0 else statistics.variance(data)
            assert table.variance(ddof) == pytest.approx(expected)
            assert table.stdev(ddof) == pytest.approx(math.sqrt(expected))
        assert table.data_range == max(data) - min(data)
        assert table.quartiles == pytest.approx(series.quartiles)
        assert table.interquartile_range == pytest.approx(series.interquartile_range)
        assert table.mean_absolute_deviation == pytest.approx(series.mean_absolute_deviation)
        assert table.median_absolute_deviation == pytest.approx(series.median_absolute_deviation)
        for p in (0, 1, 25, 50, 75, 99, 100):
            for method in ("nearest", "linear"):
                assert table.percentile(p, method=method) == pytest.approx(series.percentile(p, method=method))
        for x in (-30, 0, 1.5, 30):
            assert table.frequency(x) == data.count(x)
            assert table.pmf(x) == pytest.approx(data.count(x) / n)
            assert table.cumulative_frequency(x) == sum(v <= x for v in data)
            assert table.cdf(x) == pytest.approx(sum(v <= x for v in data) / n)


def test_categories_and_missing_values():
    table = FrequencyTable(["b", "a", "b", "c"])
    assert table.most_common() == [("b", 2), ("a", 1), ("c", 1)]
    assert table.most_common(2) == [("b", 2), ("a", 1)]
    assert table.most_common(0) == []
    assert table.pmf("missing") == 0
    assert table.relative_frequency("b") == .5
    assert table.cdf("b") == .75
    assert table.cumulative_relative_frequency("b") == .75
    assert table.quantile(.5) == "b"
    with pytest.raises(KeyError):
        _ = table["missing"]


def test_merge_order_validation_and_immutability():
    table = FrequencyTable(["b", "a", "b"])
    other = FrequencyTable.from_frequencies({"a": 3, "c": 2})
    result = table.merge(other)
    assert list(result.items()) == [("b", 2), ("a", 4), ("c", 2)]
    assert list(table.items()) == [("b", 2), ("a", 1)]
    assert dict(other) == {"a": 3, "c": 2}
    assert table.merge({}) == table
    assert table.merge({"z": 0}) == table
    with pytest.raises(ValueError):
        table.merge({"a": -1})
    with pytest.raises(TypeError):
        table.merge({"a": True})


def test_large_offset_variance():
    # Integers preserve the small offsets even beyond float exact range.
    data = [10**18 + x for x in (-2, -1, 0, 1, 2)]
    table = FrequencyTable(data)
    assert table.variance() == 2
    assert table.variance(1) == 2.5


def test_huge_counts_without_expansion():
    table = FrequencyTable.from_frequencies({0: 10**12, 10: 10**12})
    with patch.object(FrequencyTable, "iter_counts", side_effect=AssertionError("expanded")), patch.object(FrequencyTable, "to_series", side_effect=AssertionError("expanded")):
        assert table.variance() == 25
        assert table.stdev() == 5
        assert table.quartiles == (0., 5., 10.)
        assert table.mean_absolute_deviation == 5
        assert table.median_absolute_deviation == 5
        assert table.cdf(5) == .5
        assert table.coefficient_of_variation() == 1
    enormous = FrequencyTable.from_frequencies({1: 10**30, 2: 1})
    assert enormous.quantile(1) == 2
    assert enormous.quantile(1, method="linear") == 2.


@pytest.mark.parametrize("ddof", [-1, 3, 4])
def test_invalid_ddof_value(ddof):
    with pytest.raises(ValueError):
        FrequencyTable([1, 2, 3]).variance(ddof)


@pytest.mark.parametrize("ddof", [True, .5, "1", None])
def test_invalid_ddof_type(ddof):
    with pytest.raises(TypeError):
        FrequencyTable([1, 2, 3]).stdev(ddof)


@pytest.mark.parametrize("k", [-.1, 1.1, math.nan, math.inf])
def test_invalid_quantile(k):
    with pytest.raises(ValueError):
        FrequencyTable([1, 2]).quantile(k)


def test_empty_and_constant_tables():
    empty = FrequencyTable([])
    assert empty.frequency(1) == 0
    assert empty.cumulative_frequency(1) == 0
    assert empty.most_common() == []
    for call in (lambda: empty.variance(), lambda: empty.cdf(1), lambda: empty.pmf(1),
                 lambda: empty.quantile(.5), lambda: empty.quartiles,
                 lambda: empty.data_range, lambda: empty.mean_absolute_deviation,
                 lambda: empty.median_absolute_deviation):
        with pytest.raises(ValueError):
            call()
    table = FrequencyTable([5, 5])
    assert table.variance() == table.interquartile_range == 0
    assert table.quartiles == (5., 5., 5.)
    with pytest.raises(ValueError):
        FrequencyTable([-1, 1]).coefficient_of_variation()
    with pytest.raises(ValueError):
        table.quantile(.5, method="bad")


@pytest.mark.parametrize("n, exception", [(-1, ValueError), (True, TypeError), (.5, TypeError)])
def test_most_common_validation(n, exception):
    with pytest.raises(exception):
        FrequencyTable([1]).most_common(n)


@pytest.mark.parametrize("rank", [.5, True, "1"])
def test_find_pos_requires_integer(rank):
    with pytest.raises(TypeError):
        FrequencyTable([1, 2]).find_pos(rank)
