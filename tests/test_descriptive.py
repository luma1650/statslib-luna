
import random
from collections import Counter
from collections.abc import Mapping
from statistics import mean as std_mean
from statistics import median as std_median
from statistics import multimode as std_multimode

import pytest

from statslib.frequency.table import FrequencyTable
from statslib.series import StatsSeries


# ==========================================
# HELPERS
# ==========================================

def make_table(data):
    return FrequencyTable(data)


# ==========================================
# CONSTRUCTION
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([], {}),
    ([1], {1: 1}),
    ([1, 1, 2, 3, 3, 3], {1: 2, 2: 1, 3: 3}),
    (["a", "b", "a"], {"a": 2, "b": 1}),
    ((1, 2, 2), {1: 1, 2: 2}),
])
def test_construction(data, expected):
    table = make_table(data)
    assert dict(table) == expected
    assert isinstance(table, Mapping)
    assert len(table) == len(expected)
    assert table.total == sum(expected.values())


def test_generator_input():
    table = make_table(x % 3 for x in range(10))
    assert dict(table) == {0: 4, 1: 3, 2: 3}


def test_unhashable_input():
    with pytest.raises(TypeError):
        make_table([[1], [1], [2]])


# ==========================================
# MAPPING INTERFACE
# ==========================================

def test_getitem():
    table = make_table([2, 2, 3, 3, 3, 5])
    assert table[2] == 2
    assert table[3] == 3
    assert table[5] == 1


def test_missing_key():
    table = make_table([1, 2])
    with pytest.raises(KeyError):
        _ = table[99]


def test_mapping_methods():
    table = make_table([2, 2, 3, 3, 3, 5])

    assert table.get(99, 0) == 0
    assert 3 in table
    assert 99 not in table
    assert list(table.keys()) == [2, 3, 5]
    assert list(table.values()) == [2, 3, 1]
    assert list(table.items()) == [(2, 2), (3, 3), (5, 1)]


def test_repr():
    assert repr(make_table([1, 1, 2])) == (
        "FrequencyTable({1: 2, 2: 1})"
    )


# ==========================================
# FROM FREQUENCIES
# ==========================================

def test_from_frequencies():
    table = FrequencyTable.from_frequencies({
        2: 3, 4: 2, 7: 1
    })
    assert dict(table) == {2: 3, 4: 2, 7: 1}
    assert table.total == 6


def test_zero_frequencies_removed():
    table = FrequencyTable.from_frequencies({
        1: 0, 2: 3, 3: 0
    })
    assert dict(table) == {2: 3}


@pytest.mark.parametrize("frequency", [-1, -10])
def test_negative_frequency(frequency):
    with pytest.raises(ValueError):
        FrequencyTable.from_frequencies({1: frequency})


@pytest.mark.parametrize("frequency", [
    1.5, "3", None, True, False
])
def test_invalid_frequency_type(frequency):
    with pytest.raises(TypeError):
        FrequencyTable.from_frequencies({1: frequency})


def test_compact_storage():
    table = FrequencyTable.from_frequencies({
        1: 1_000_000_000,
        2: 2_000_000_000,
    })

    assert len(table) == 2
    assert table.total == 3_000_000_000


# ==========================================
# IMMUTABILITY
# ==========================================

def test_absolute_immutable():
    table = make_table([1, 1, 2])
    with pytest.raises(TypeError):
        table.absolute[1] = 100


def test_relative_immutable():
    table = make_table([1, 1, 2])
    with pytest.raises(TypeError):
        table.relative[1] = 0.99


def test_input_independence():
    data = [1, 1, 2]
    table = make_table(data)
    data.append(3)
    assert dict(table) == {1: 2, 2: 1}


# ==========================================
# ITER_COUNTS / TO_SERIES
# ==========================================

def test_iter_counts():
    table = FrequencyTable.from_frequencies({
        2: 3, 4: 2, 7: 1
    })
    assert list(table.iter_counts()) == [
        2, 2, 2, 4, 4, 7
    ]


def test_iter_counts_lazy():
    table = FrequencyTable.from_frequencies({
        1: 1_000_000_000,
        2: 1_000_000_000,
    })
    iterator = table.iter_counts()
    assert next(iterator) == 1
    assert next(iterator) == 1


def test_to_series():
    table = make_table([2, 2, 3, 3, 3, 5])
    result = table.to_series()

    assert isinstance(result, StatsSeries)
    assert list(result) == [2, 2, 3, 3, 3, 5]


def test_roundtrip():
    data = [5, 2, 3, 2, 5, 3, 1]
    table = make_table(data)
    restored = table.to_series()

    assert Counter(restored) == Counter(data)


# ==========================================
# RELATIVE AND PERCENTAGE
# ==========================================

def test_relative():
    table = make_table([2, 2, 3, 3, 3, 5])

    assert table.relative[2] == pytest.approx(2 / 6)
    assert table.relative[3] == pytest.approx(3 / 6)
    assert table.relative[5] == pytest.approx(1 / 6)
    assert sum(table.relative.values()) == pytest.approx(1)


def test_percentage():
    table = make_table([2, 2, 3, 3, 3, 5])
    result = table.percentage()

    assert result[2] == pytest.approx(100 / 3)
    assert result[3] == pytest.approx(50)
    assert result[5] == pytest.approx(100 / 6)
    assert sum(result.values()) == pytest.approx(100)


# ==========================================
# CUMULATIVE
# ==========================================

def test_cumulative():
    table = make_table([2, 2, 3, 3, 3, 5])
    assert dict(table.cumulative()) == {
        2: 2, 3: 5, 5: 6
    }


def test_cumulative_reverse():
    table = make_table([2, 2, 3, 3, 3, 5])
    assert dict(table.cumulative(reverse=True)) == {
        5: 1, 3: 4, 2: 6
    }


def test_cumulative_unsorted_input():
    table = make_table([5, 3, 2, 3, 2, 3])
    assert dict(table.cumulative()) == {
        2: 2, 3: 5, 5: 6
    }


def test_cumulative_custom_key():
    table = make_table([
        "Alto", "Bajo", "Medio",
        "Alto", "Medio", "Alto"
    ])
    order = {"Bajo": 0, "Medio": 1, "Alto": 2}

    result = table.cumulative(key=order.__getitem__)

    assert dict(result) == {
        "Bajo": 1,
        "Medio": 3,
        "Alto": 6,
    }


def test_cumulative_relative():
    table = make_table([2, 2, 3, 3, 3, 5])
    result = table.cumulative_relative()

    assert result[2] == pytest.approx(2 / 6)
    assert result[3] == pytest.approx(5 / 6)
    assert result[5] == pytest.approx(1)


def test_cumulative_relative_reverse():
    table = make_table([2, 2, 3, 3, 3, 5])
    result = table.cumulative_relative(reverse=True)

    assert result[5] == pytest.approx(1 / 6)
    assert result[3] == pytest.approx(4 / 6)
    assert result[2] == pytest.approx(1)


def test_empty_frequencies():
    table = make_table([])

    assert table.total == 0
    assert dict(table.relative) == {}
    assert dict(table.percentage()) == {}
    assert dict(table.cumulative()) == {}
    assert dict(table.cumulative_relative()) == {}


# ==========================================
# MEAN / MEDIAN / MODE / MULTIMODE
# ==========================================

@pytest.mark.parametrize("data", [
    [1],
    [1, 2, 3],
    [1, 2, 3, 4],
    [2, 2, 3, 3, 3, 5],
    [-10, -5, 0, 5, 10],
    [1.5, 2.5, 3.5],
    [5, 5, 5, 5],
])
def test_mean_against_stdlib(data):
    table = make_table(data)
    assert table.mean == pytest.approx(std_mean(data))


@pytest.mark.parametrize("data", [
    [1],
    [1, 2],
    [1, 2, 3],
    [1, 2, 3, 4],
    [1, 1, 1, 2, 3],
    [5, 2, 9, 2, 1, 5],
    [-10, -5, 0, 5],
    [5, 5, 5, 5],
])
def test_median_against_stdlib(data):
    table = make_table(data)
    assert table.median == pytest.approx(std_median(data))


def test_median_large_counts():
    table = FrequencyTable.from_frequencies({
        10: 1_000_000,
        20: 2_000_000,
        30: 1_000_000,
    })
    assert table.median == pytest.approx(20)


def test_median_unsorted_frequencies():
    table = FrequencyTable.from_frequencies({
        30: 2, 10: 2, 20: 2
    })
    assert table.median == pytest.approx(20)


def test_mode():
    table = make_table([1, 2, 2, 3])
    assert table.mode == 2


def test_mode_categorical():
    table = make_table(["red", "blue", "red"])
    assert table.mode == "red"


def test_multimode():
    table = make_table([1, 1, 2, 2, 3])
    assert set(table.multimode) == {1, 2}


def test_multimode_single():
    table = make_table([1, 2, 2, 3])
    assert list(table.multimode) == [2]


def test_multimode_all_tied():
    table = make_table([1, 2, 3])
    assert set(table.multimode) == {1, 2, 3}


# ==========================================
# RANDOMIZED REFERENCE TESTS
# ==========================================

def test_randomized_statistics():
    rng = random.Random(42)

    for n in range(1, 101):
        for _ in range(20):
            data = [
                rng.randint(-25, 25)
                for _ in range(n)
            ]
            table = make_table(data)

            assert table.total == len(data)
            assert dict(table) == dict(Counter(data))
            assert table.mean == pytest.approx(std_mean(data))
            assert table.median == pytest.approx(std_median(data))
            assert set(table.multimode) == set(std_multimode(data))


# ==========================================
# CACHING
# ==========================================

def test_cached_properties():
    table = make_table([1, 1, 2, 3])

    assert table.relative is table.relative
    assert table.mean is table.mean
    assert table.median is table.median


def test_cached_methods():
    table = make_table([1, 1, 2, 3])

    assert table.cumulative() is table.cumulative()
    assert table.cumulative(reverse=True) is table.cumulative(reverse=True)
    assert table.percentage() == table.percentage()
    assert table.to_series() == table.to_series()
