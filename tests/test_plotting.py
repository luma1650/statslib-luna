"""Check actual rendered artists and weighted statistical equivalence."""
import builtins
import subprocess
import sys
from unittest.mock import patch

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from statslib import FrequencyTable, StatsSeries


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_line_preserves_observation_order_and_style():
    series = StatsSeries([5, 1, 3])
    _, ax = plt.subplots()
    assert series.plot(ax=ax, color="red", label="sample") is ax
    line = ax.lines[0]
    assert list(line.get_xdata()) == [0, 1, 2]
    assert list(line.get_ydata()) == [5, 1, 3]
    assert line.get_color() == "red"
    assert line.get_label() == "sample"
    assert list(series) == [5, 1, 3]


@pytest.mark.parametrize("density", [False, True])
def test_weighted_histogram_matches_expanded_sample(density):
    data = [1, 1, 1, 2, 4, 4]
    bins = [0, 2, 3, 5]
    series_ax = StatsSeries(data).hist(bins, density=density)
    table = FrequencyTable(data)
    with patch.object(FrequencyTable, "iter_counts", side_effect=AssertionError("expanded")):
        table_ax = table.hist(bins, density=density)
    expected, _ = np.histogram(data, bins, density=density)
    assert [p.get_height() for p in series_ax.patches] == pytest.approx(expected)
    assert [p.get_height() for p in table_ax.patches] == pytest.approx(expected)


def test_categorical_bars_keep_distinct_positions_and_order():
    table = FrequencyTable([1, "1", "1", "blue"])
    ax = table.bar(relative=True, color="green")
    assert [p.get_height() for p in ax.patches] == [.25, .5, .25]
    assert [t.get_text() for t in ax.get_xticklabels()] == ["1", "1", "blue"]
    assert len({p.get_x() for p in ax.patches}) == 3
    _, other = plt.subplots()
    assert table.plot(ax=other) is other
    assert [p.get_height() for p in other.patches] == [1, 2, 1]


@pytest.mark.parametrize("complementary", [False, True])
def test_weighted_ecdf_matches_expanded_sample(complementary):
    data = [4, 1, 1, 1, 2, 4]
    a = StatsSeries(data).ecdf(complementary=complementary, compress=True)
    b = FrequencyTable(data).ecdf(complementary=complementary, compress=True)
    np.testing.assert_allclose(a.lines[0].get_xdata(), b.lines[0].get_xdata())
    np.testing.assert_allclose(a.lines[0].get_ydata(), b.lines[0].get_ydata())
    expected = [1, .5, 1 / 3, 0] if complementary else [0, .5, 2 / 3, 1]
    np.testing.assert_allclose(a.lines[0].get_ydata(), expected, atol=1e-15)


def test_huge_frequencies_are_not_expanded():
    table = FrequencyTable.from_frequencies({1: 10**12, 2: 2 * 10**12})
    with patch.object(FrequencyTable, "iter_counts", side_effect=AssertionError("expanded")):
        ax = table.hist([0, 1.5, 3])
        assert [p.get_height() for p in ax.patches] == [10**12, 2 * 10**12]
        ecdf = table.ecdf()
        assert ecdf.lines[0].get_ydata()[-1] == pytest.approx(1)


def test_boxplot_median_and_axes_reuse():
    _, ax = plt.subplots()
    assert StatsSeries([1, 2, 3, 4, 9]).boxplot(ax=ax) is ax
    assert list(ax.lines[5].get_ydata()) == [3, 3]


@pytest.mark.parametrize("cls, methods", [
    (StatsSeries, ["plot", "hist", "boxplot", "ecdf"]),
    (FrequencyTable, ["plot", "bar", "hist", "ecdf"]),
])
def test_empty_data_does_not_create_figures(cls, methods):
    for method in methods:
        with pytest.raises(ValueError, match="empty"):
            getattr(cls([]), method)()
    assert plt.get_fignums() == []


def test_table_rejects_weight_override_and_auto_bins():
    table = FrequencyTable([1, 2])
    for method in (table.hist, table.ecdf):
        with pytest.raises(TypeError, match="weights"):
            method(weights=[1, 1])
    with pytest.raises(ValueError, match="bin"):
        table.hist("auto")
    assert plt.get_fignums() == []


def test_plotting_never_calls_show():
    with patch.object(plt, "show", side_effect=AssertionError("unexpected show")):
        ax = StatsSeries([1, 2, 3]).hist()
        ax.figure.canvas.draw()


def test_missing_optional_dependency_has_install_hint():
    from statslib.plotting import get_axes
    real_import = builtins.__import__
    def without_matplotlib(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("not installed")
        return real_import(name, *args, **kwargs)
    with patch("builtins.__import__", side_effect=without_matplotlib):
        with pytest.raises(ImportError, match=r"statslib-luna\[plot\]"):
            get_axes()


def test_statistics_import_does_not_load_matplotlib():
    result = subprocess.run(
        [sys.executable, "-c", "import statslib, sys; assert 'matplotlib' not in sys.modules; assert statslib.StatsSeries([1,2,3]).mean == 2"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_horizontal_histogram_labels():
    for data in (StatsSeries([1, 2, 2]), FrequencyTable([1, 2, 2])):
        ax = data.hist(orientation="horizontal", density=True)
        assert ax.get_xlabel() == "Density"
        assert ax.get_ylabel() == "Value"


def test_horizontal_ecdf_and_supplied_weights():
    ax = StatsSeries([1, 1, 2]).ecdf(
        orientation="horizontal", weights=[1, 2, 3], compress=True,
    )
    np.testing.assert_allclose(ax.lines[0].get_xdata(), [0, .5, 1])
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [1, 1, 2])
