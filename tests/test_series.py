
import pytest

from collections.abc import Sequence
from unittest.mock import patch

from statslib.series import StatsSeries


# ==========================================
# CONSTRUCTION
# ==========================================

@pytest.mark.parametrize("data", [
    [],
    [1],
    [1, 2, 3],
    (1, 2, 3),
    range(10),
    ["a", "b", "c"],
])
def test_construction(data):
    series = StatsSeries(data)

    assert list(series) == list(data)
    assert len(series) == len(data)


def test_generator_construction():
    series = StatsSeries(x * 2 for x in range(5))

    assert list(series) == [0, 2, 4, 6, 8]


def test_original_list_mutation():
    data = [1, 2, 3]
    series = StatsSeries(data)

    data.append(4)
    data[0] = 100

    assert list(series) == [1, 2, 3]


def test_sequence_interface():
    series = StatsSeries([1, 2, 3])

    assert isinstance(series, Sequence)


# ==========================================
# LENGTH, ITERATION AND REVERSE
# ==========================================

def test_len():
    assert len(StatsSeries([1, 2, 3])) == 3
    assert len(StatsSeries([])) == 0


def test_iteration():
    series = StatsSeries([1, 2, 3])

    assert list(iter(series)) == [1, 2, 3]
    assert list(iter(series)) == [1, 2, 3]


def test_reversed():
    series = StatsSeries([1, 2, 3, 4])

    assert list(reversed(series)) == [4, 3, 2, 1]


def test_reversed_empty():
    assert list(reversed(StatsSeries([]))) == []


# ==========================================
# INDEXING
# ==========================================

@pytest.mark.parametrize("index, expected", [
    (0, 10),
    (1, 20),
    (2, 30),
    (-1, 30),
    (-2, 20),
    (-3, 10),
])
def test_getitem(index, expected):
    series = StatsSeries([10, 20, 30])

    assert series[index] == expected


@pytest.mark.parametrize("index", [3, -4, 100])
def test_getitem_out_of_range(index):
    series = StatsSeries([10, 20, 30])

    with pytest.raises(IndexError):
        _ = series[index]


# ==========================================
# SLICING
# ==========================================

@pytest.mark.parametrize("slc, expected", [
    (slice(1, 4), [2, 3, 4]),
    (slice(None, None), [1, 2, 3, 4, 5]),
    (slice(None, None, 2), [1, 3, 5]),
    (slice(None, None, -1), [5, 4, 3, 2, 1]),
    (slice(2, 2), []),
    (slice(10, 20), []),
    (slice(-3, None), [3, 4, 5]),
])
def test_slicing(slc, expected):
    series = StatsSeries([1, 2, 3, 4, 5])

    result = series[slc]

    assert isinstance(result, StatsSeries)
    assert list(result) == expected


def test_slicing_returns_independent_series():
    series = StatsSeries([1, 2, 3, 4])

    subset = series[1:3]

    assert subset is not series
    assert list(subset) == [2, 3]


def test_slice_zero_step():
    series = StatsSeries([1, 2, 3])

    with pytest.raises(ValueError):
        _ = series[::0]


# ==========================================
# SEQUENCE MIXINS
# ==========================================

def test_contains():
    series = StatsSeries([1, 2, 3])

    assert 2 in series
    assert 10 not in series


def test_count():
    series = StatsSeries([2, 2, 3, 4, 2])

    assert series.count(2) == 3
    assert series.count(10) == 0


def test_index():
    series = StatsSeries([5, 3, 5, 7])

    assert series.index(5) == 0
    assert series.index(7) == 3

    with pytest.raises(ValueError):
        series.index(100)


# ==========================================
# EQUALITY
# ==========================================

def test_equality():
    a = StatsSeries([1, 2, 3])
    b = StatsSeries([1, 2, 3])
    c = StatsSeries([3, 2, 1])

    assert a == b
    assert a != c


def test_equality_different_types():
    series = StatsSeries([1, 2, 3])

    assert series != [1, 2, 3]
    assert series != (1, 2, 3)


def test_equality_empty():
    assert StatsSeries([]) == StatsSeries([])


# ==========================================
# HASHING
# ==========================================

def test_hash_equal_series():
    a = StatsSeries([1, 2, 3])
    b = StatsSeries([1, 2, 3])

    assert hash(a) == hash(b)


def test_hash_matches_tuple():
    series = StatsSeries([1, 2, 3])

    assert hash(series) == hash((1, 2, 3))


def test_hash_as_dict_key():
    cache = {
        StatsSeries([1, 2, 3]): "dataset A"
    }

    assert cache[StatsSeries([1, 2, 3])] == "dataset A"


def test_unhashable_elements():
    series = StatsSeries([[1], [2]])

    with pytest.raises(TypeError):
        hash(series)


# ==========================================
# FROM FREQUENCIES
# ==========================================

def test_from_frequencies():
    series = StatsSeries.from_frequencies({
        2: 3,
        4: 2,
        7: 1,
    })

    assert list(series) == [2, 2, 2, 4, 4, 7]


def test_from_frequencies_zero():
    series = StatsSeries.from_frequencies({
        1: 0,
        2: 2,
        3: 0,
    })

    assert list(series) == [2, 2]


def test_from_frequencies_empty():
    series = StatsSeries.from_frequencies({})

    assert len(series) == 0


@pytest.mark.parametrize("frequency", [
    -1,
    -100,
])
def test_from_frequencies_negative(frequency):
    with pytest.raises(ValueError):
        StatsSeries.from_frequencies({1: frequency})


@pytest.mark.parametrize("frequency", [
    1.5,
    "3",
    None,
    True,
    False,
])
def test_from_frequencies_invalid_type(frequency):
    with pytest.raises(TypeError):
        StatsSeries.from_frequencies({1: frequency})


def test_from_frequencies_string_keys():
    series = StatsSeries.from_frequencies({
        "a": 2,
        "b": 1,
    })

    assert list(series) == ["a", "a", "b"]


# ==========================================
# REPRESENTATION
# ==========================================

def test_repr():
    series = StatsSeries([1, 2, 3])

    assert repr(series) == "StatsSeries([1, 2, 3])"


def test_repr_empty():
    assert repr(StatsSeries([])) == "StatsSeries([])"


# ==========================================
# STATISTICAL PROPERTIES
# ==========================================

def test_mean():
    series = StatsSeries([2, 4, 6, 8])

    assert series.mean == pytest.approx(5.0)


def test_median_odd():
    series = StatsSeries([5, 1, 3])

    assert series.median == pytest.approx(3.0)


def test_median_even():
    series = StatsSeries([1, 2, 3, 4])

    assert series.median == pytest.approx(2.5)


def test_mode():
    series = StatsSeries([1, 2, 2, 3])

    assert series.mode == 2


def test_multimode():
    series = StatsSeries([1, 1, 2, 2, 3])

    assert set(series.multimode) == {1, 2}


def test_weighted_mean():
    series = StatsSeries([10, 20, 30])

    result = series.weighted_mean([1, 2, 3])

    assert result == pytest.approx(140 / 6)


def test_statistical_properties_on_slice():
    series = StatsSeries([10, 20, 30, 40, 50])

    subset = series[1:4]

    assert subset.mean == pytest.approx(30)
    assert subset.median == pytest.approx(30)


# ==========================================
# CACHED PROPERTIES
# ==========================================

def test_mean_is_cached():
    series = StatsSeries([1, 2, 3])

    with patch("statslib.series.mean", return_value=2.0) as mocked:
        assert series.mean == 2.0
        assert series.mean == 2.0

        mocked.assert_called_once()


def test_median_is_cached():
    series = StatsSeries([1, 2, 3])

    with patch("statslib.series.median", return_value=2.0) as mocked:
        assert series.median == 2.0
        assert series.median == 2.0

        mocked.assert_called_once()


def test_different_instances_have_separate_caches():
    a = StatsSeries([1, 2, 3])
    b = StatsSeries([10, 20, 30])

    assert a.mean == 2
    assert b.mean == 20


# ==========================================
# EMPTY SERIES
# ==========================================

def test_empty_mean():
    series = StatsSeries([])

    with pytest.raises(ValueError):
        _ = series.mean


def test_empty_median():
    series = StatsSeries([])

    with pytest.raises(ValueError):
        _ = series.median


def test_empty_mode():
    series = StatsSeries([])

    with pytest.raises(ValueError):
        _ = series.mode
