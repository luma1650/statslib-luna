"""Presentation rows must retain counts, ordering and safe rendering."""
from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest

from statslib import FrequencyTable, GroupedFrequencyTable, FrequencyRow, GroupedFrequencyRow
from statslib.models.interval import Interval


def test_frequency_row_metrics_and_immutability():
    row = FrequencyRow("a", 2, 3, 4)
    assert row.relative_frequency == .5
    assert row.cumulative_relative_frequency == .75
    assert row.percentage == 50
    assert row.cumulative_percentage == 75
    assert row.to_dict()["Value"] == "a"
    assert "50.0000" in str(row)
    with pytest.raises(FrozenInstanceError):
        row.frequency = 10
    data = row.to_dict()
    data["Frequency"] = 99
    assert row.frequency == 2


def test_rows_follow_requested_order():
    table = FrequencyTable([3, 1, 3, 2])
    rows = table.rows()
    assert isinstance(rows, tuple)
    assert [row.value for row in rows] == [3, 1, 2]
    assert [row.cumulative_frequency for row in rows] == [2, 3, 4]
    assert [row.value for row in table.rows(sort=True)] == [1, 2, 3]
    assert [row.cumulative_frequency for row in table.rows(sort=True)] == [1, 2, 4]
    assert [row.value for row in table.rows(reverse=True)] == [3, 2, 1]
    assert [row.value for row in table.rows(key=lambda value: -value)] == [3, 2, 1]
    assert list(table) == [3, 1, 2]


def test_mixed_categories_require_no_sorting():
    table = FrequencyTable([1, "a", (2, 3)])
    assert len(table.rows()) == 3
    assert "(2, 3)" in table.to_text()
    with pytest.raises(TypeError):
        table.rows(sort=True)


def test_grouped_rows_interval_features_and_zeros():
    intervals = [Interval(0, 2), Interval(2, 5), Interval(5, 10, closed_right=True)]
    table = GroupedFrequencyTable([0, 1, 5, 10], intervals)
    rows = table.rows()
    assert all(isinstance(row, GroupedFrequencyRow) for row in rows)
    assert [row.frequency for row in rows] == [2, 0, 2]
    assert [row.cumulative_frequency for row in rows] == [2, 2, 4]
    assert [row.midpoint for row in rows] == [1, 3.5, 7.5]
    assert [row.width for row in rows] == [2, 3, 5]
    assert [row.density for row in rows] == [1, 0, .4]
    assert sum(row.relative_frequency for row in rows) == 1
    assert rows[-1].cumulative_relative_frequency == 1
    assert rows[-1].to_dict()["Interval"] == intervals[-1]
    with pytest.raises(FrozenInstanceError):
        rows[0].interval = intervals[1]


def test_empty_displays():
    table = FrequencyTable([])
    assert table.rows() == ()
    assert "Frequency" in table.to_text()
    assert "<tbody></tbody>" in table._repr_html_()
    grouped = GroupedFrequencyTable([], [Interval(0, 1)])
    row, = grouped.rows()
    assert row.frequency == row.relative_frequency == row.cumulative_percentage == 0
    assert "0.0000" in grouped.to_text()
    assert FrequencyRow("empty", 0, 0, 0).relative_frequency == 0


def test_html_escapes_values_and_formats_headers():
    row = FrequencyRow('<script>alert("x")</script>&', 1, 1, 1)
    html = row._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "&amp;" in html
    assert "<th>Frequency</th>" in html
    assert "<script>" not in FrequencyTable([row.value])._repr_html_()
    grouped = GroupedFrequencyTable([1], [Interval(0, 2)])
    assert "<th>Midpoint</th>" in grouped._repr_html_()


def test_text_precision_and_column_alignment():
    table = FrequencyTable(["long value", "b", "b"])
    text = table.to_text(precision=2)
    assert "0.33" in text
    positions = [[i for i, c in enumerate(line) if c == "|"] for line in text.splitlines() if "|" in line]
    assert all(p == positions[0] for p in positions)
    assert "0.3333" not in text


@pytest.mark.parametrize("precision, error", [(-1, ValueError), (True, TypeError), (1.5, TypeError)])
def test_invalid_precision(precision, error):
    with pytest.raises(error):
        FrequencyTable([1]).to_text(precision=precision)


@pytest.mark.parametrize("counts, error", [
    ((-1, 0, 1), ValueError), ((2, 1, 3), ValueError),
    ((1, 3, 2), ValueError), ((True, 1, 1), TypeError),
    ((1, 1.5, 2), TypeError), ((1, 1, "2"), TypeError),
])
@pytest.mark.parametrize("grouped", [False, True])
def test_invalid_row_counts(counts, error, grouped):
    cls = GroupedFrequencyRow if grouped else FrequencyRow
    value = Interval(0, 1) if grouped else "a"
    with pytest.raises(error):
        cls(value, *counts)


def test_grouped_row_rejects_non_interval():
    with pytest.raises(TypeError):
        GroupedFrequencyRow("[0, 1)", 1, 1, 1)


def test_huge_counts_do_not_expand():
    table = FrequencyTable.from_frequencies({1: 10**20, 2: 10**20})
    with patch.object(FrequencyTable, "iter_counts", side_effect=AssertionError("expanded")), patch.object(FrequencyTable, "to_series", side_effect=AssertionError("expanded")):
        assert table.rows()[-1].cumulative_frequency == 2 * 10**20
        assert "200000000000000000000" in table.to_text()
        assert "<table>" in table._repr_html_()
