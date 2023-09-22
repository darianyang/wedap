"""
Unit and regression tests for the MD_Pdist class.
"""

# Import package, test suite, and other packages as needed
import mdap

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

# TODO: test for trace, search_aux, skip_basis, get_total_data_array, get_all_weights
# maybe test more args like first_iter, last_iter, step_iter, H5save_out, data_proc, bins, histrange, p_units

class Test_MD_Pdist():
    """
    Test each method of the MD_Pdist class.
    """
    data_path = "mdap/tests/data/"
    
    @pytest.mark.parametrize("Xname", ["input_data_0.dat", "input_data_2.npy", "input_data_2.pkl"])
    def test_timeseries(self, Xname):
        X, Y, _ = mdap.MD_Pdist(Xname=self.data_path + Xname, data_type="time").pdist()

        # X data is just the frames / timescale
        np.testing.assert_allclose(X, np.loadtxt(f"mdap/tests/data/timeseries_{Xname}_X.txt"))

        # Y data is the data value at each frame
        np.testing.assert_allclose(Y, np.loadtxt(f"mdap/tests/data/timeseries_{Xname}_Y.txt"))
        
    @pytest.mark.parametrize("Xname", ["input_data_3.dat"])
    @pytest.mark.parametrize("Xindex", [1, 2])
    @pytest.mark.parametrize("Xinterval", [1, 10])
    def test_pdist_1d(self, Xname, Xindex, Xinterval):
        X, Y, _ = mdap.MD_Pdist(data_type="pdist", Xname=self.data_path + Xname, 
                                Xindex=Xindex, Xinterval=Xinterval).pdist()
        np.testing.assert_allclose(X, np.loadtxt(f"mdap/tests/data/pdist1d_{Xname}_idx{Xindex}_int{Xinterval}_X.txt"))
        np.testing.assert_allclose(Y, np.loadtxt(f"mdap/tests/data/pdist1d_{Xname}_idx{Xindex}_int{Xinterval}_Y.txt"))

    @pytest.mark.parametrize("Xname", ["input_data_0.dat", "input_data_2.npy", "input_data_2.pkl"])
    @pytest.mark.parametrize("Yname", ["input_data_3.dat"])
    @pytest.mark.parametrize("Yindex", [1, 2])
    def test_pdist_2d(self, Xname, Yname, Yindex):
        X, Y, Z = mdap.MD_Pdist(data_type="pdist", Xname=self.data_path + Xname, Xinterval=10,
                                Yname=self.data_path + Yname, Yindex=Yindex).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"mdap/tests/data/pdist2d_{Xname}_{Yname}_yidx{Yindex}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"mdap/tests/data/pdist2d_{Xname}_{Yname}_yidx{Yindex}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"mdap/tests/data/pdist2d_{Xname}_{Yname}_yidx{Yindex}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["input_data_0.dat"])
    @pytest.mark.parametrize("Yname", ["input_data_1.dat", "input_data_2.npy"])
    @pytest.mark.parametrize("Zname", ["input_data_2.dat", "input_data_2.pkl"])
    def test_pdist_3d(self, Xname, Yname, Zname):
        X, Y, Z = mdap.MD_Pdist(data_type="pdist", Xname=self.data_path + Xname, 
                                Yname=self.data_path + Yname, Zname=self.data_path + Zname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"mdap/tests/data/pdist3d_{Xname}_{Yname}_{Zname}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"mdap/tests/data/pdist3d_{Xname}_{Yname}_{Zname}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"mdap/tests/data/pdist3d_{Xname}_{Yname}_{Zname}_Z.txt"))
