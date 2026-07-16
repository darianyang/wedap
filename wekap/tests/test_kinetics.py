"""
Unit and regression tests for the wekap Kinetics class.

Uses the small direct.h5 / assign.h5 files checked into wekap/data.
"""

import os

import numpy as np
import pytest

import matplotlib
matplotlib.use('agg')

import wekap

# paths to the checked-in sample data (relative to repo root, matching the
# other test suites which are run from the repo root)
DATA = os.path.join(os.path.dirname(__file__), os.pardir, "data")
DIRECT = os.path.join(DATA, "direct.h5")
MULTI_DIRECT = [os.path.join(DATA, f) for f in ("direct.h5", "direct2.h5", "direct3.h5")]

# a realistic tau (100 ps) so rates come out in sensible s^-1 units
TAU = 100e-12


class Test_Kinetics:
    """Test the core Kinetics methods on the sample direct.h5 file."""

    def test_extract_rate_shape_and_finite(self):
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1)
        rate, ci_lb, ci_ub = k.extract_rate()
        # rate evolution should be 1D and match the CI array lengths
        assert rate.ndim == 1
        assert rate.shape == ci_lb.shape == ci_ub.shape
        # the converged (final) rate should be finite and positive
        assert np.isfinite(rate[-1])
        assert rate[-1] > 0

    def test_concentration_scaling(self):
        """Dividing by concentration should scale the rate by exactly that factor."""
        k1 = wekap.Kinetics(DIRECT, tau=TAU, state=1, concentration=1)
        k10 = wekap.Kinetics(DIRECT, tau=TAU, state=1, concentration=10)
        r1 = k1.extract_rate()[0]
        r10 = k10.extract_rate()[0]
        # compare where finite and nonzero
        mask = np.isfinite(r1) & np.isfinite(r10) & (r10 != 0)
        np.testing.assert_allclose(r1[mask] / r10[mask], 10.0, rtol=1e-6)

    @pytest.mark.parametrize("statepop", ["direct", "assign"])
    def test_statepop_options(self, statepop):
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1, statepop=statepop)
        rate = k.extract_rate()[0]
        assert np.all(np.isfinite(rate))

    @pytest.mark.parametrize("state", [0, 1])
    def test_state_selection(self, state):
        k = wekap.Kinetics(DIRECT, tau=TAU, state=state)
        rate = k.extract_rate()[0]
        assert rate.ndim == 1
        assert np.isfinite(rate[-1])

    @pytest.mark.parametrize("x_units", ["iterations", "moltime", "agg"])
    def test_x_units(self, x_units):
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1, x_units=x_units)
        rate = k.extract_rate()[0]
        x = k._get_x_data(len(rate))
        assert x.shape[0] == rate.shape[0]

    def test_flux_units_mfpts(self):
        """mfpts should be the reciprocal of the rate."""
        k_rate = wekap.Kinetics(DIRECT, tau=TAU, state=1, flux_units="rates")
        rate = k_rate.plot_rate()
        # rates and mfpts share the same underlying extract_rate
        assert np.isfinite(rate[-1]) and rate[-1] > 0

    def test_red_scheme(self):
        """The RED-corrected rate should run and give finite, non-negative rates."""
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1, red=True, red_timepoints=101)
        rate, ci_lb, ci_ub = k.extract_rate()
        assert rate.shape == ci_lb.shape == ci_ub.shape
        assert np.all(np.isfinite(rate))
        # no negative rates (early iterations may be exactly zero before durations accrue)
        assert np.all(rate >= 0)
        # the converged RED rate should be positive
        assert rate[-1] > 0

    def test_red_correction_factor(self):
        """The RED correction factors should be finite and non-negative."""
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1, red=True, red_timepoints=101)
        n_iters = k.direct_h5["rate_evolution"].shape[0]
        corr = k._red_correction(n_iters, timepoints=101)
        assert corr.shape[0] == n_iters
        assert np.all(np.isfinite(corr))
        assert np.all(corr >= 0)

    def test_red_timepoints_autodetect(self):
        """With assign.h5 present, red_timepoints=None should auto-detect from npts."""
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1, red=True, red_timepoints=None)
        # should not raise and should return a sensible integer
        tp = k._get_red_timepoints()
        assert isinstance(tp, int)
        assert tp >= 2

    def test_plot_rate_runs(self):
        k = wekap.Kinetics(DIRECT, tau=TAU, state=1)
        rate = k.plot_rate()
        assert np.isfinite(rate[-1])
        matplotlib.pyplot.close("all")

    def test_plot_multi_rates(self):
        k = wekap.Kinetics(MULTI_DIRECT[0], tau=TAU, state=1)
        x, multi_k, multi_k_avg, multi_k_unc = k.plot_multi_rates(MULTI_DIRECT, plotting=False)
        # one rate array per replicate
        assert len(multi_k) == len(MULTI_DIRECT)
        # averaged rate matches the x-axis length
        assert multi_k_avg.shape[0] == x.shape[0]
        matplotlib.pyplot.close("all")
