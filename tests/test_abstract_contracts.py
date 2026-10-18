"""Exercise ABC defaults through independent storage implementations."""
from collections.abc import ItemsView, KeysView, Mapping, ValuesView
import inspect

import pytest

from statslib.abc import AbstractFrequencyTable, AbstractStatsSeries
from statslib import FrequencyTable, GroupedFrequencyTable, StatsSeries
from statslib.models.interval import Interval


class DictTable(AbstractFrequencyTable):
    """Minimal mutable implementation: only the three Mapping primitives."""
    def __init__(self, counts):
        self.data = dict(counts)

    def __getitem__(self, key):
        return self.data[key]

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)


class ListSeries(AbstractStatsSeries):
    def __init__(self, data):
        self.data = list(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        value = self.data[index]
        return type(self)(value) if isinstance(index, slice) else value


@pytest.mark.parametrize("cls", [AbstractFrequencyTable, AbstractStatsSeries])
def test_bases_cannot_be_instantiated(cls):
    assert inspect.isabstract(cls)
    with pytest.raises(TypeError):
        cls()


@pytest.mark.parametrize("table", [DictTable({10: 2, 20: 3}), FrequencyTable([10, 10, 20, 20, 20]),
    GroupedFrequencyTable([1, 1, 3], [Interval(0, 2), Interval(2, 4)])])
def test_mapping_views_and_concrete_inheritance(table):
    assert isinstance(table, AbstractFrequencyTable)
    assert isinstance(table, Mapping)
    assert not inspect.isabstract(type(table))
    assert isinstance(table.keys(), KeysView)
    assert isinstance(table.values(), ValuesView)
    assert isinstance(table.items(), ItemsView)
    assert list(table.frequencies()) == list(table.values())
    assert sum(table.values()) == table.total
    assert dict(table.items()) == dict(table)
    assert len(table) == len(table.keys())
    assert sum(table.percentage().values()) == pytest.approx(100)
    assert table.mode == table.multimode[0]
    assert list(table.iter_counts()).count(next(iter(table))) == table[next(iter(table))]


@pytest.mark.parametrize("cls", [DictTable, FrequencyTable])
def test_default_and_optimized_implementations_agree(cls):
    table = cls({3: 2, 1: 1, 2: 2}) if cls is DictTable else cls([3, 3, 1, 2, 2])
    assert table.mode == 3
    assert table.multimode == (3, 2)
    assert table.frequency(999) == table.pmf(999) == 0
    assert table.relative_frequency(3) == .4
    assert table.cdf(2) == table.cumulative_relative_frequency(2) == .6
    assert table.most_common(2) == [(3, 2), (2, 2)]
    assert table.percentage() == {3: 40., 1: 20., 2: 40.}
    with pytest.raises(KeyError):
        _ = table[999]


def test_defaults_recompute_for_mutable_subclasses():
    table = DictTable({1: 1, 2: 1})
    snapshot = table.relative
    assert table.total == 2
    assert table.multimode == (1, 2)
    table.data[2] = 3
    assert table.total == 4
    assert table.relative[2] == .75
    assert table.mode == 2
    assert snapshot[2] == .5
    with pytest.raises(TypeError):
        table.absolute[2] = 9


@pytest.mark.parametrize("table", [DictTable({}), FrequencyTable([]), GroupedFrequencyTable([], [Interval(0, 1)])])
def test_no_observations_contract(table):
    assert table.total == 0
    # Mapping truth is based on keys: configured grouped bins are still keys.
    assert bool(table) == (len(table) != 0)
    assert table.multimode == ()
    assert all(count == 0 for count in table.percentage().values())
    with pytest.raises(ValueError):
        _ = table.mode
    with pytest.raises(ValueError):
        table.pmf(next(iter(table), 0))


def test_grouped_cdf_uses_interval_order_and_counts():
    bins = [Interval(0, 2), Interval(2, 4), Interval(4, 6)]
    table = GroupedFrequencyTable([0, 1, 5], bins)
    assert table.cdf(bins[1]) == pytest.approx(2 / 3)
    assert table.cumulative_relative_frequency(bins[2]) == 1
    assert table.mode == bins[0]
    assert table.multimode == table.modal_intervals == (bins[0],)
    assert table.pmf(Interval(10, 20)) == 0
    assert list(table.values()) == [2, 0, 1]
    with pytest.raises(KeyError):
        table.cdf(Interval(10, 20))


@pytest.mark.parametrize("ddof", [True, 1.5, None])
def test_grouped_ddof_types_match_series_and_table(ddof):
    grouped = GroupedFrequencyTable([1, 3], [Interval(0, 2), Interval(2, 4)])
    for obj in (grouped, FrequencyTable([1, 3]), StatsSeries([1, 3])):
        with pytest.raises(TypeError):
            obj.variance(ddof)


def test_series_defaults_use_sequence_interface_and_remain_fresh():
    series = ListSeries([1, 2, 3])
    assert series.mean == 2
    assert series.variance() == pytest.approx(2 / 3)
    assert series.quartiles == (1.5, 2., 2.5)
    assert dict(series.frequency_table) == {1: 1, 2: 1, 3: 1}
    assert list(series[1:]) == [2, 3]
    series.data.append(6)
    assert series.mean == 3
    assert list(ListSeries.from_frequencies({1: 2, 3: 1})) == [1, 1, 3]


def test_plotting_is_available_to_independent_series_implementations():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = ListSeries([3, 1, 1, 2])
    try:
        for method in ("plot", "hist", "boxplot", "ecdf"):
            ax = getattr(series, method)()
            ax.figure.canvas.draw()
        assert list(series) == [3, 1, 1, 2]
    finally:
        plt.close("all")
