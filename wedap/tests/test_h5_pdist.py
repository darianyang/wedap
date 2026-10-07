"""
Unit and regression tests for the H5_Pdist class.
"""

# Import package, test suite, and other packages as needed
import wedap

import h5py
import shutil
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


def _assert_writable(h5_path):
    """
    Opening for writing fails if another handle (e.g. a leaked H5_Pdist.h5) is still open.
    """
    with h5py.File(h5_path, "a"):
        pass

class Test_H5_Pdist_File_Handling():
    """
    The h5 file should not be left open (and locked) after errors or explicit closing.
    """
    @pytest.fixture
    def h5_copy(self, tmp_path):
        h5_path = tmp_path / "p53.h5"
        shutil.copyfile("wedap/data/p53.h5", h5_path)
        return str(h5_path)

    def test_closed_after_init_error(self, h5_copy):
        # keep the exception (and its traceback) alive, like an interactive session would
        with pytest.raises(ValueError, match="last_iter") as excinfo:
            wedap.H5_Pdist(h5=h5_copy, data_type="evolution", last_iter=10_000)
        _assert_writable(h5_copy)
        assert excinfo is not None

    def test_closed_after_plot_init_error(self, h5_copy):
        with pytest.raises(ValueError, match="not a valid object") as excinfo:
            wedap.H5_Plot(h5=h5_copy, data_type="evolution", Xname="not_a_dataset")
        _assert_writable(h5_copy)
        assert excinfo is not None

    def test_context_manager(self, h5_copy):
        with wedap.H5_Pdist(h5=h5_copy, data_type="evolution") as pdist:
            pdist.pdist()
        _assert_writable(h5_copy)
        # closing again is a no-op
        pdist.close()

    def test_h5_save_out_closed(self, h5_copy, tmp_path):
        out = str(tmp_path / "saved.h5")
        with wedap.H5_Pdist(h5=h5_copy, data_type="evolution", last_iter=5,
                            H5save_out=out, Xsave_name="pcoord_copy") as pdist:
            pdist.pdist()
        assert pdist.H5save_out == out
        _assert_writable(out)
        with h5py.File(out, "r") as f:
            assert "iterations/iter_00000005/auxdata/pcoord_copy" in f

    def test_h5_save_out_same_as_input(self, h5_copy):
        pdist = wedap.H5_Pdist(h5=h5_copy, data_type="evolution", last_iter=5,
                               H5save_out=h5_copy, Xsave_name="pcoord_copy")
        with pdist, pytest.raises(ValueError, match="must be different"):
            pdist.pdist()

    def test_h5_save_out_existing_save_name(self, h5_copy, tmp_path):
        out = tmp_path / "saved.h5"
        pdist = wedap.H5_Pdist(h5=h5_copy, data_type="evolution", last_iter=5,
                               H5save_out=str(out), Xsave_name="dihedral_2")
        with pdist, pytest.raises(ValueError, match="already exists"):
            pdist.pdist()
        assert not out.exists()

    def test_h5_save_out_multiple_h5_warns(self, h5_copy, tmp_path):
        h5_copy_2 = str(tmp_path / "p53_2.h5")
        shutil.copyfile(h5_copy, h5_copy_2)
        out = str(tmp_path / "saved.h5")
        with wedap.H5_Pdist(h5=[h5_copy, h5_copy_2], data_type="evolution", last_iter=5,
                            H5save_out=out, Xsave_name="pcoord_copy") as pdist:
            with pytest.warns(UserWarning, match="only the first file"):
                pdist.pdist()

    def test_h5_save_out_failure_keeps_existing_output(self, h5_copy, tmp_path, monkeypatch):
        out = tmp_path / "saved.h5"
        out.write_bytes(b"previous output")
        # fail partway through writing the new datasets
        get_data_array = wedap.H5_Pdist._get_data_array
        def failing_get_data_array(self, name, index, iteration, h5_create=None, h5_create_name=None):
            if h5_create is not None and iteration == 3:
                raise RuntimeError("simulated failure")
            return get_data_array(self, name, index, iteration, h5_create, h5_create_name)
        monkeypatch.setattr(wedap.H5_Pdist, "_get_data_array", failing_get_data_array)

        pdist = wedap.H5_Pdist(h5=h5_copy, data_type="evolution", last_iter=5,
                               H5save_out=str(out), Xsave_name="pcoord_copy")
        with pdist, pytest.raises(RuntimeError, match="simulated failure"):
            pdist.pdist()
        # existing output untouched and no leftover temp file
        assert out.read_bytes() == b"previous output"
        assert sorted(p.name for p in tmp_path.iterdir()) == ["p53.h5", "saved.h5"]

class Test_Succ_Only_Weights():
    """
    succ_only weights should line up with the iterations from first_iter to last_iter.
    p53.h5 has no recycling events, so w_succ is stubbed with a few walkers to trace back.
    """
    h5 = "wedap/data/p53.h5"
    succ = [(12, 0), (14, 2)]

    def _stub_w_succ(self, monkeypatch):
        succ = self.succ
        monkeypatch.setattr(wedap.H5_Pdist, "w_succ", lambda pdist: succ)

    def _succ_weights(self, monkeypatch, **kwargs):
        self._stub_w_succ(monkeypatch)
        with wedap.H5_Pdist(h5=self.h5, data_type="average", last_iter=15,
                            no_pbar=True, **kwargs) as pdist:
            return pdist.succ_pdist_weight_filter()

    def test_first_iter(self, monkeypatch):
        ref = self._succ_weights(monkeypatch, first_iter=1)
        weights = self._succ_weights(monkeypatch, first_iter=5)
        assert len(weights) == 11
        for iteration in range(5, 16):
            np.testing.assert_array_equal(weights[iteration - 5], ref[iteration - 1])
        # the traced walkers keep their weight
        assert np.count_nonzero(weights[12 - 5]) > 0

    def test_step_iter(self, monkeypatch):
        ref = self._succ_weights(monkeypatch, first_iter=1)
        weights = self._succ_weights(monkeypatch, first_iter=1, step_iter=2)
        # step_iter is applied when indexing the weights, not to the weight array
        assert len(weights) == len(ref) == 15
        for w, r in zip(weights, ref):
            np.testing.assert_array_equal(w, r)

    def test_multiple_h5(self, monkeypatch, tmp_path):
        # two copies of the same file should give the same succ_only pdist as one file
        h5_copy = str(tmp_path / "p53_copy.h5")
        shutil.copyfile(self.h5, h5_copy)
        self._stub_w_succ(monkeypatch)
        def pdist(h5, succ_only):
            with wedap.H5_Pdist(h5=h5, data_type="average", last_iter=15,
                                succ_only=succ_only, no_pbar=True) as pdist:
                return pdist.pdist()[1]
        single = pdist(self.h5, succ_only=True)
        multi = pdist([self.h5, h5_copy], succ_only=True)
        np.testing.assert_allclose(multi, single)
        # and the filter should actually change the result
        assert not np.allclose(multi, pdist(self.h5, succ_only=False))

    def test_h5_save_out(self, monkeypatch, tmp_path):
        out = str(tmp_path / "succ.h5")
        self._stub_w_succ(monkeypatch)
        with wedap.H5_Pdist(h5=self.h5, data_type="average", first_iter=5, last_iter=15,
                            succ_only=True, H5save_out=out, no_pbar=True) as pdist:
            pdist.pdist()
            weights = pdist.weights
        with h5py.File(self.h5, "r") as f_in, h5py.File(out, "r") as f_out:
            # weights before first_iter are untouched
            for iteration in range(1, 5):
                path = f"iterations/iter_{iteration:08d}/seg_index"
                np.testing.assert_array_equal(f_out[path]["weight"], f_in[path]["weight"])
            for iteration in range(5, 16):
                path = f"iterations/iter_{iteration:08d}/seg_index"
                np.testing.assert_array_equal(f_out[path]["weight"], weights[iteration - 5])

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