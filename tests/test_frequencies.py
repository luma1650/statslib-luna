
import pytest

from collections.abc import Mapping

from statslib.frequency.table import FrequencyTable


# ==========================================
# CONSTRUCTION
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([1, 1, 2, 3, 3, 3], {1: 2, 2: 1, 3: 3}),
    ([5, 5, 5], {5: 3}),
    ([1], {1: 1}),
    ([], {}),
    (["a", "b", "a"], {"a": 2, "b": 1}),
    ((1, 2, 2), {1: 1, 2: 2}),
])
def test_construction(data, expected):
    table = FrequencyTable(data)

    assert dict(table) == expected


def test_generator_construction():
    table = FrequencyTable(x % 3 for x in range(10))

    assert dict(table) == {
        0: 4,
        1: 3,
        2: 3,
    }


def test_mapping_interface():
    table = FrequencyTable([1, 1, 2])

    assert isinstance(table, Mapping)


# ==========================================
# GETITEM
# ==========================================

def test_getitem():
    table = FrequencyTable([2, 2, 3, 3, 3, 5])

    assert table[2] == 2
    assert table[3] == 3
    assert table[5] == 1


def test_getitem_missing():
    table = FrequencyTable([1, 2, 2])

    with pytest.raises(KeyError):
        _ = table[999]


def test_get_default():
    table = FrequencyTable([1, 2, 2])

    assert table.get(2, 0) == 2
    assert table.get(999, 0) == 0


# ==========================================
# ITERATION AND LENGTH
# ==========================================

def test_iteration():
    table = FrequencyTable([3, 1, 3, 2])

    assert list(table) == [3, 1, 2]


def test_len_counts_distinct_values():
    table = FrequencyTable([2, 2, 3, 3, 3, 5])

    assert len(table) == 3


def test_len_empty():
    assert len(FrequencyTable([])) == 0


def test_membership():
    table = FrequencyTable([1, 1, 2])

    assert 1 in table
    assert 3 not in table


def test_keys_values_items():
    table = FrequencyTable([2, 2, 3])

    assert list(table.keys()) == [2, 3]
    assert list(table.values()) == [2, 1]
    assert list(table.items()) == [(2, 2), (3, 1)]


# ==========================================
# ABSOLUTE FREQUENCY
# ==========================================

def test_absolute():
    table = FrequencyTable([2, 2, 3, 3, 3, 5])

    assert dict(table.absolute) == {
        2: 2,
        3: 3,
        5: 1,
    }


def test_absolute_immutable():
    table = FrequencyTable([1, 1, 2])

    with pytest.raises(TypeError):
        table.absolute[1] = 100


def test_absolute_empty():
    assert dict(FrequencyTable([]).absolute) == {}


# ==========================================
# TOTAL
# ==========================================

@pytest.mark.parametrize("data, expected", [
    ([], 0),
    ([1], 1),
    ([1, 1, 2], 3),
    ([5] * 100, 100),
    (range(1000), 1000),
])
def test_total(data, expected):
    table = FrequencyTable(data)

    assert table.total == expected


def test_total_is_cached():
    table = FrequencyTable([1, 1, 2])

    first = table.total
    second = table.total

    assert first == second == 3
    assert "total" in table.__dict__


# ==========================================
# RELATIVE FREQUENCY
# ==========================================

def test_relative():
    table = FrequencyTable([2, 2, 3, 3, 3, 5])

    assert table.relative[2] == pytest.approx(2 / 6)
    assert table.relative[3] == pytest.approx(3 / 6)
    assert table.relative[5] == pytest.approx(1 / 6)


def test_relative_sum():
    table = FrequencyTable([1, 1, 2, 3, 3, 3])

    assert sum(table.relative.values()) == pytest.approx(1.0)


def test_relative_empty():
    table = FrequencyTable([])

    assert dict(table.relative) == {}


def test_relative_immutable():
    table = FrequencyTable([1, 1, 2])

    with pytest.raises(TypeError):
        table.relative[1] = 0.99


def test_relative_is_cached():
    table = FrequencyTable([1, 1, 2])

    first = table.relative
    second = table.relative

    assert first is second


# ==========================================
# FROM FREQUENCIES
# ==========================================

def test_from_frequencies():
    table = FrequencyTable.from_frequencies({
        2: 3,
        4: 2,
        7: 1,
    })

    assert dict(table) == {2: 3, 4: 2, 7: 1}
    assert table.total == 6


def test_from_frequencies_empty():
    table = FrequencyTable.from_frequencies({})

    assert dict(table) == {}
    assert table.total == 0


def test_from_frequencies_zero():
    table = FrequencyTable.from_frequencies({
        1: 0,
        2: 3,
        3: 0,
    })

    assert dict(table) == {2: 3}
    assert table.total == 3


@pytest.mark.parametrize("frequency", [-1, -10, -100])
def test_from_frequencies_negative(frequency):
    with pytest.raises(ValueError):
        FrequencyTable.from_frequencies({1: frequency})


@pytest.mark.parametrize("frequency", [
    1.5,
    "3",
    None,
    True,
    False,
])
def test_from_frequencies_invalid_type(frequency):
    with pytest.raises(TypeError):
        FrequencyTable.from_frequencies({1: frequency})


def test_from_frequencies_large():
    table = FrequencyTable.from_frequencies({
        1: 1_000_000,
        2: 2_000_000,
    })

    assert len(table) == 2
    assert table.total == 3_000_000
    assert table.relative[1] == pytest.approx(1 / 3)


def test_from_frequencies_does_not_mutate_input():
    data = {1: 2, 2: 3}
    original = data.copy()

    FrequencyTable.from_frequencies(data)

    assert data == original


# ==========================================
# UNHASHABLE DATA
# ==========================================

def test_unhashable_values():
    with pytest.raises(TypeError):
        FrequencyTable([[1], [1], [2]])


# ==========================================
# REPRESENTATION
# ==========================================

def test_repr():
    table = FrequencyTable([1, 1, 2])

    assert repr(table) == "FrequencyTable({1: 2, 2: 1})"


def test_repr_empty():
    assert repr(FrequencyTable([])) == "FrequencyTable({})"


# ==========================================
# INTEGRATION
# ==========================================

def test_from_statsseries():
    from statslib.series import StatsSeries

    series = StatsSeries([2, 2, 3, 3, 3, 5])
    table = FrequencyTable(series)

    assert dict(table) == {2: 2, 3: 3, 5: 1}


def test_series_frequency_table():
    from statslib.series import StatsSeries

    series = StatsSeries([1, 1, 2, 3])
    table = series.frequency_table

    assert isinstance(table, FrequencyTable)
    assert dict(table) == {1: 2, 2: 1, 3: 1}
