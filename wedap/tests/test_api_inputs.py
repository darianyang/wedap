"""
Tests for feeding numpy arrays directly into the wedap API and for forwarding
matplotlib artist kwargs (e.g. alpha) from the main class to the plot methods.
"""

import numpy as np
import pytest

import matplotlib
matplotlib.use('agg')

import wedap

H5 = "wedap/data/p53.h5"


@pytest.fixture(scope="module")
def xyz():
    """Precompute a 2D pdist once to reuse as raw X/Y/Z array input."""
    return wedap.H5_Pdist(H5, "average", Yname="pcoord").pdist()


@pytest.mark.parametrize("cls_name", ["H5_Plot", "Plot"])
def test_precomputed_arrays_plot(xyz, cls_name):
    """Precomputed X/Y/Z arrays should plot directly via H5_Plot or the Plot alias."""
    X, Y, Z = xyz
    cls = getattr(wedap, cls_name)
    plot = cls(X, Y, Z, plot_mode="hist")
    plot.plot()
    # arrays should be used as-is (no pdist recomputation)
    np.testing.assert_array_equal(plot.Z, Z)
    matplotlib.pyplot.close("all")


def test_pdist_plot_aliases_are_classes():
    """wedap.Pdist / wedap.Plot should alias the H5 classes."""
    assert wedap.Pdist is wedap.H5_Pdist
    assert wedap.Plot is wedap.H5_Plot


def test_alpha_forwarded_to_hist(xyz):
    """An mpl artist kwarg (alpha) should reach the pcolormesh QuadMesh."""
    X, Y, Z = xyz
    plot = wedap.H5_Plot(X, Y, Z, plot_mode="hist", alpha=0.4)
    plot.plot()
    assert plot.plot_obj.get_alpha() == 0.4
    matplotlib.pyplot.close("all")


def test_alpha_forwarded_to_line():
    """An mpl artist kwarg (alpha) should reach a 1D line plot."""
    plot = wedap.H5_Plot(h5=H5, data_type="average", plot_mode="line", alpha=0.3)
    plot.plot(cbar=False)
    line = plot.ax.get_lines()[0]
    assert line.get_alpha() == 0.3
    matplotlib.pyplot.close("all")
