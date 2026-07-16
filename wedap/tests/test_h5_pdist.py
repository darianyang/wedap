"""
Unit and regression tests for the H5_Pdist class.
"""

# Import package, test suite, and other packages as needed
import wedap

import numpy as np
import pytest

# look at file coverage for testing
# pytest -v --cov=wedap
# produces .coverage binary file to be used by other tools to visualize 
# do not need 100% coverage, 80-90% is very high

# can have report in 
# $ pytest -v --cov=wedap --cov-report=html
# index.html to better visualize the test coverage

# decorator to skip in pytest
#@pytest.mark.skip


def assert_close(actual, desired, rtol=1e-5, atol=1e-4,
                 max_bad_frac=1e-3, max_bad_abs=0.05):
    """
    Compare pdist arrays with a tolerance appropriate for float32-derived data
    that is robust across numpy versions.

    Two numpy-version effects motivate this:
      1. The default assert_allclose rtol of 1e-7 is too tight -- numpy 2 changed
         some reduction/accumulation orderings, shifting histogram bin centers at
         the float32-precision level (~1e-6 relative).
      2. A data point sitting exactly on a histogram bin boundary can land in a
         different bin under a different numpy, flipping the probability of a
         couple of bins (amplified by the -ln(P) transform).

    So require the bulk of elements to match within (rtol, atol), while allowing a
    very small fraction of isolated bins to differ -- but those outliers must still
    be small in absolute terms, so gross regressions (which flip many bins or shift
    values a lot) still fail.
    """
    a = np.asarray(actual, dtype=float)
    d = np.asarray(desired, dtype=float)

    # empty histogram bins become inf under -ln(P); inf/inf positions that agree
    # are fine (assert_allclose treats inf==inf as equal). Only measure amplitude
    # where both are finite; count finiteness disagreements as outliers.
    both_finite = np.isfinite(a) & np.isfinite(d)
    finiteness_mismatch = np.isfinite(a) != np.isfinite(d)

    absdiff = np.zeros(d.shape, dtype=float)
    absdiff[both_finite] = np.abs(a[both_finite] - d[both_finite])
    tol = atol + rtol * np.where(np.isfinite(d), np.abs(d), 0.0)

    bad = (absdiff > tol) | finiteness_mismatch
    bad_frac = float(np.mean(bad)) if bad.size else 0.0
    max_diff = float(absdiff[both_finite].max()) if both_finite.any() else 0.0

    assert bad_frac <= max_bad_frac, \
        f"{bad_frac:.4%} of elements exceed tolerance (max finite abs diff {max_diff:.4g})"
    assert max_diff <= max_bad_abs, \
        f"max abs diff {max_diff:.4g} exceeds {max_bad_abs}"


# TODO: test for trace, search_aux, skip_basis, get_total_data_array, get_all_weights
# maybe test more args like first_iter, last_iter, step_iter, H5save_out, data_proc, bins, histrange, p_units
# could also change to 1/2/3 dataset format

class Test_H5_Pdist():
    """
    Test each method of the H5_Pdist class.
    """
    h5 = "wedap/data/p53.h5"
    
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_evolution(self, Xname):
        evolution = wedap.H5_Pdist(h5=self.h5, data_type="evolution", Xname=Xname)
        X, Y, Z = evolution.pdist()

        # X data is the variably filled array of instance pdist x values
        assert_close(X, np.loadtxt(f"wedap/tests/data/evolution_{Xname}_X.txt"))

        # Y data is just the WE iterations
        assert_close(Y, 
            np.arange(evolution.first_iter, evolution.last_iter + 1, 1))

        # Z data is the pdist values of each iteration
        assert_close(Z, np.loadtxt(f"wedap/tests/data/evolution_{Xname}_Z.txt"))

    # this repeat test is needed since I want to test both pcoord vs aux and multiple indices
    @pytest.mark.parametrize("Xname", ["pcoord"])
    @pytest.mark.parametrize("Xindex", [0, 1])
    def test_evolution_idx(self, Xname, Xindex):
        evolution = wedap.H5_Pdist(h5=self.h5, data_type="evolution", Xname=Xname, Xindex=Xindex)
        X, Y, Z = evolution.pdist()

        # X data is the variably filled array of instance pdist x values
        assert_close(X, np.loadtxt(f"wedap/tests/data/evolution_{Xname}{Xindex}_X.txt"))

        # Y data is just the WE iterations
        assert_close(Y, 
            np.arange(evolution.first_iter, evolution.last_iter + 1, 1))

        # Z data is the pdist values of each iteration
        assert_close(Z, np.loadtxt(f"wedap/tests/data/evolution_{Xname}{Xindex}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_instant_1d(self, Xname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xname=Xname).pdist()
        assert_close(X, np.loadtxt(f"wedap/tests/data/instant_{Xname}_X.txt"))
        assert_close(Y, np.loadtxt(f"wedap/tests/data/instant_{Xname}_Y.txt"))
        
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    @pytest.mark.parametrize("Yname", ["dihedral_3", "dihedral_4"])
    def test_instant_2d(self, Xname, Yname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xname=Xname, Yname=Yname).pdist()
        assert_close(X, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}_{Yname}_X.txt"))
        assert_close(Y, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}_{Yname}_Y.txt"))
        assert_close(Z, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}_{Yname}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord"])
    #@pytest.mark.parametrize("Yname", ["dihedral_3", "pcoord"])
    @pytest.mark.parametrize("Yname", ["dihedral_3"])
    @pytest.mark.parametrize("Xindex", [0, 1])
    def test_instant_2d_idx(self, Xname, Yname, Xindex):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xindex=Xindex,
                                 Xname=Xname, Yname=Yname).pdist()
        assert_close(X, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_X.txt"))
        assert_close(Y, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_Y.txt"))
        assert_close(Z, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_Z.txt"))
    
    # TODO along with average 3D (but this is kinda taken care of in H5_Plot scatter3d tests)
    # def test_instant_3d(self):
    #     X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xname=Xname, Yname=Yname).pdist()
    #     np.testing.assert_allclose(X, 
    #         np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_X.txt"))
    #     np.testing.assert_allclose(Y, 
    #         np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_Y.txt"))
    #     np.testing.assert_allclose(Z, 
    #         np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_average_1d(self, Xname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="average", Xname=Xname).pdist()
        assert_close(X, np.loadtxt(f"wedap/tests/data/average_{Xname}_X.txt"))
        assert_close(Y, np.loadtxt(f"wedap/tests/data/average_{Xname}_Y.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    @pytest.mark.parametrize("Yname", ["dihedral_3", "dihedral_4"])
    def test_average_2d(self, Xname, Yname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="average", Xname=Xname, Yname=Yname).pdist()
        assert_close(X, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}_X.txt"))
        assert_close(Y, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}_Y.txt"))
        assert_close(Z, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}_Z.txt"))
        
    #@pytest.mark.parametrize("Xname", ["dihedral_3", "pcoord"])
    @pytest.mark.parametrize("Xname", ["dihedral_3"])
    @pytest.mark.parametrize("Yname", ["pcoord"])
    @pytest.mark.parametrize("Yindex", [0, 1])
    def test_average_2d_idx(self, Xname, Yname, Yindex):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="average", Yindex=Yindex,
                                 Xname=Xname, Yname=Yname).pdist()
        assert_close(X, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_X.txt"))
        assert_close(Y, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_Y.txt"))
        assert_close(Z, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_Z.txt"))