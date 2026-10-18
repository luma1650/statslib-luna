"""Regression tests and reference checks for statistical extensions."""
import math
import random
import statistics

import pytest

from statslib import StatsSeries, FrequencyTable, WelfordMoments
from statslib.core.descriptive import multimode, quickselect
from statslib.core.moments import origin_moment, central_moment, standard_moment
from statslib.core.quantiles import quantile, quantile_linear


def test_cumulative_orders_are_independent():
    table = FrequencyTable([3, 1, 1, 2])
    assert list(table.cumulative().items()) == [(1, 2), (2, 3), (3, 4)]
    assert list(table.cumulative(reverse=True).items()) == [(3, 1), (2, 2), (1, 4)]
    assert list(table.cumulative(key=lambda x: -x).items()) == [(3, 1), (2, 2), (1, 4)]
    assert table.find_pos(0) == 1
    assert table.find_pos(0, reverse=True) == 3
    assert table.median == 1.5
    assert table.cumulative() is table.cumulative()


def test_custom_sort_does_not_poison_median():
    table = FrequencyTable([10, 2, 2, 3])
    table.cumulative(key=str)
    assert table.median == 2.5


@pytest.mark.parametrize("data, expected", [
    ([[1]], [[1]]),
    ([[2], [1]], [[1], [2]]),
    ([[2], [2], [1], [1]], [[1], [2]]),
    ([{"x": 1}, {"x": 1}, {"x": 2}], [{"x": 1}]),
    ([{"x": 1}, {"x": 2}], [{"x": 1}, {"x": 2}]),
])
def test_unhashable_multimode(data, expected):
    assert multimode(data) == expected


@pytest.mark.parametrize("data", [list(range(3000)), list(range(3000, 0, -1)), [7] * 10000])
def test_quickselect_structured_inputs(data):
    original = data.copy()
    for k in (0, 1, len(data) // 2, len(data) - 1):
        assert quickselect(data.copy(), k) == sorted(data)[k]
    assert data == original


def test_quantiles_randomized_reference():
    rng = random.Random(80)
    for n in range(2, 80):
        data = [rng.uniform(-100, 100) for _ in range(n)]
        reference = statistics.quantiles(data, n=100, method="inclusive")
        series = StatsSeries(data)
        for p in (1, 25, 50, 75, 99):
            assert series.percentile(p, method="linear") == pytest.approx(reference[p - 1])
            assert quantile(iter(data), p / 100) == sorted(data)[math.ceil(p / 100 * n) - 1]
        assert series.quartiles == pytest.approx((reference[24], reference[49], reference[74]))
        assert series.interquartile_range == pytest.approx(reference[74] - reference[24])


@pytest.mark.parametrize("k", [-.1, 1.1, math.nan, math.inf])
@pytest.mark.parametrize("function", [quantile, quantile_linear])
def test_quantile_invalid_probability(function, k):
    with pytest.raises(ValueError):
        function([1, 2], k)


def test_series_empty_and_constant_cases():
    with pytest.raises(ValueError):
        _ = StatsSeries([]).quartiles
    with pytest.raises(ValueError):
        _ = StatsSeries([2, 2]).skewness
    with pytest.raises(ValueError):
        StatsSeries([-1, 1]).coefficient_of_variation()
    with pytest.raises(ValueError):
        StatsSeries([1]).quantile(.5, method="invalid")
    assert StatsSeries([4]).quartiles == (4., 4., 4.)


@pytest.mark.parametrize("function", [origin_moment, central_moment, standard_moment])
@pytest.mark.parametrize("order", [True, 1.5, "2", None])
def test_moment_order_type(function, order):
    with pytest.raises(TypeError):
        function([1, 2, 3], order)


def test_moments_reference_and_streaming():
    rng = random.Random(42)
    for n in range(2, 80):
        data = [rng.uniform(-20, 20) for _ in range(n)]
        series = StatsSeries(data)
        mu = statistics.mean(data)
        v = statistics.pvariance(data)
        stream = WelfordMoments()
        stream.extend(iter(data))
        for order in range(5):
            assert series.origin_moment(order) == pytest.approx(sum(x ** order for x in data) / n, rel=1e-12, abs=1e-12)
            assert series.central_moment(order) == pytest.approx(sum((x - mu) ** order for x in data) / n, rel=1e-12, abs=1e-12)
        skew = sum((x - mu) ** 3 for x in data) / n / v ** 1.5
        kurtosis = sum((x - mu) ** 4 for x in data) / n / v ** 2 - 3
        assert series.skewness == pytest.approx(skew, rel=1e-12, abs=1e-12)
        assert series.excess_kurtosis == pytest.approx(kurtosis, rel=1e-12, abs=1e-12)
        assert stream.variance(0) == pytest.approx(v)
        assert series.coefficient_of_variation() == pytest.approx(math.sqrt(v) / abs(mu))
