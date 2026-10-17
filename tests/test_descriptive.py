
import pytest
from statistics import mean as std_mean
from statistics import median as std_median

from statslib.core.descriptive import (
    mean,
    median,
    mode,
    multimode,
    weighted_mean,
)


# ==========================================
# MEAN
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1, 2, 3, 4, 5], 3.0),
    ([10], 10.0),
    ([-5, 0, 5], 0.0),
    ([1.5, 2.5, 3.5], 2.5),
    ([0, 0, 0], 0.0),
])
def test_mean(data, expected):
    assert mean(data) == pytest.approx(expected)


def test_mean_generator():
    assert mean(x for x in range(10)) == pytest.approx(4.5)


def test_mean_empty():
    with pytest.raises(ValueError):
        mean([])


# ==========================================
# MEDIAN
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1], 1),
    ([1, 2], 1.5),
    ([1, 2, 3], 2),
    ([1, 2, 3, 4], 2.5),
    ([4, 1, 3, 2], 2.5),
    ([9, 8, 7, 6, 5], 7),
    ([5, 5, 5, 5], 5),
    ([-10, -5, 0, 5], -2.5),
    ([1.5, 2.5, 3.5], 2.5),
])
def test_median(data, expected):
    assert median(data) == pytest.approx(expected)


def test_median_generator():
    assert median(x for x in [5, 1, 3, 2, 4]) == 3


def test_median_empty():
    with pytest.raises(ValueError):
        median([])


def test_median_does_not_mutate():
    data = [5, 1, 4, 2, 3]
    original = data.copy()

    median(data)

    assert data == original


# ==========================================
# MODE
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1, 2, 2, 3], 2),
    ([5, 5, 5], 5),
    (["a", "b", "a"], "a"),
    ([1], 1),
    ([1.5, 2.5, 1.5], 1.5),
])
def test_mode(data, expected):
    assert mode(data) == expected


def test_mode_empty():
    with pytest.raises(ValueError):
        mode([])


def test_mode_generator():
    assert mode(x for x in [1, 2, 2, 3]) == 2


def test_mode_unhashable_orderable():
    data = [[2], [1], [2], [3]]
    assert mode(data) == [2]


def test_mode_unhashable_equality_only():
    class Item:
        __hash__ = None

        def __init__(self, value):
            self.value = value

        def __eq__(self, other):
            if not isinstance(other, Item):
                return NotImplemented
            return self.value == other.value

    data = [Item(1), Item(2), Item(2)]
    assert mode(data).value == 2


# ==========================================
# MULTIMODE
# ==========================================

def test_multimode_single():
    assert multimode([1, 2, 2, 3]) == [2]


def test_multimode_multiple():
    assert set(multimode([1, 1, 2, 2, 3])) == {1, 2}


def test_multimode_all_equal_frequency():
    assert set(multimode([1, 2, 3])) == {1, 2, 3}


def test_multimode_empty():
    assert multimode([]) == []


# ==========================================
# WEIGHTED MEAN
# ==========================================

@pytest.mark.parametrize("data, weights, expected", [
    ([10, 20, 30], [1, 1, 1], 20.0),
    ([10, 20, 30], [1, 2, 3], 140 / 6),
    ([5], [10], 5.0),
    ([0, 100], [3, 1], 25.0),
    ([2, 4, 6], [0, 0, 1], 6.0),
])
def test_weighted_mean(data, weights, expected):
    assert weighted_mean(data, weights) == pytest.approx(expected)


def test_weighted_mean_mismatched_lengths():
    with pytest.raises(ValueError):
        weighted_mean([1, 2, 3], [1, 2])


def test_weighted_mean_zero_weights():
    with pytest.raises(ValueError):
        weighted_mean([1, 2], [0, 0])


# ==========================================
# REFERENCE TESTS
# ==========================================

@pytest.mark.parametrize("data", [
    [1, 2, 3],
    [5, 1, 8, 2, 9, 3],
    [-10, 20, 0, 40, -5],
    [7] * 20,
    list(range(100)),
    list(range(100, 0, -1)),
])
def test_mean_against_stdlib(data):
    assert mean(data) == pytest.approx(std_mean(data))


@pytest.mark.parametrize("data", [
    [1, 2, 3],
    [5, 1, 8, 2, 9, 3],
    [-10, 20, 0, 40, -5],
    [7] * 20,
    list(range(100)),
    list(range(100, 0, -1)),
])
def test_median_against_stdlib(data):
    assert median(data) == pytest.approx(std_median(data))
