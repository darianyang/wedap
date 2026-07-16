"""
Tests for feeding numpy arrays directly into the mdap API, including the
two-step MD_Pdist().pdist() -> MD_Plot(X=, Y=, Z=) handoff.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('agg')

import mdap


@pytest.fixture(scope="module")
def data_1d():
    rng = np.random.default_rng(0)
    return rng.normal(size=5000), rng.normal(size=5000)


def test_1d_ndarray_fed_directly(data_1d):
    """
    A plain 1D numpy array (single column) fed directly should work without the
    user having to also set Xindex/Yindex=0 (default index of 1 assumes a leading
    frame column from .dat files).
    """
    x, y = data_1d
    X, Y, Z = mdap.MD_Pdist(data_type="pdist", Xname=x, Yname=y).pdist()
    assert Z is not None
    assert np.asarray(Z).ndim == 2


def test_two_step_handoff(data_1d):
    """MD_Pdist().pdist() output should feed straight into MD_Plot(X=, Y=, Z=)."""
    x, y = data_1d
    X, Y, Z = mdap.MD_Pdist(data_type="pdist", Xname=x, Yname=y).pdist()
    plot = mdap.MD_Plot(X=X, Y=Y, Z=Z, plot_mode="hist")
    plot.plot()
    # precomputed arrays should be used as-is (no pdist recomputation)
    np.testing.assert_array_equal(plot.Z, Z)
    matplotlib.pyplot.close("all")


def test_md_plot_alias_and_precompute(data_1d):
    """mdap.Plot alias should also accept precomputed arrays."""
    x, y = data_1d
    X, Y, Z = mdap.Pdist(data_type="pdist", Xname=x, Yname=y).pdist()
    assert mdap.Plot is mdap.MD_Plot
    plot = mdap.Plot(X=X, Y=Y, Z=Z, plot_mode="hist")
    plot.plot()
    matplotlib.pyplot.close("all")


def test_1d_pdist_single_array(data_1d):
    """A single 1D array should produce a 1D pdist without index errors."""
    x, _ = data_1d
    X, Y, Z = mdap.MD_Pdist(data_type="pdist", Xname=x).pdist()
    assert X is not None and Y is not None
