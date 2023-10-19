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
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/tests/data/evolution_{Xname}_X.txt"))

        # Y data is just the WE iterations
        np.testing.assert_allclose(Y, 
            np.arange(evolution.first_iter, evolution.last_iter + 1, 1))

        # Z data is the pdist values of each iteration
        np.testing.assert_allclose(Z, np.loadtxt(f"wedap/tests/data/evolution_{Xname}_Z.txt"))

    # this repeat test is needed since I want to test both pcoord vs aux and multiple indices
    @pytest.mark.parametrize("Xname", ["pcoord"])
    @pytest.mark.parametrize("Xindex", [0, 1])
    def test_evolution_idx(self, Xname, Xindex):
        evolution = wedap.H5_Pdist(h5=self.h5, data_type="evolution", Xname=Xname, Xindex=Xindex)
        X, Y, Z = evolution.pdist()

        # X data is the variably filled array of instance pdist x values
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/tests/data/evolution_{Xname}{Xindex}_X.txt"))

        # Y data is just the WE iterations
        np.testing.assert_allclose(Y, 
            np.arange(evolution.first_iter, evolution.last_iter + 1, 1))

        # Z data is the pdist values of each iteration
        np.testing.assert_allclose(Z, np.loadtxt(f"wedap/tests/data/evolution_{Xname}{Xindex}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_instant_1d(self, Xname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xname=Xname).pdist()
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/tests/data/instant_{Xname}_X.txt"))
        np.testing.assert_allclose(Y, np.loadtxt(f"wedap/tests/data/instant_{Xname}_Y.txt"))
        
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    @pytest.mark.parametrize("Yname", ["dihedral_3", "dihedral_4"])
    def test_instant_2d(self, Xname, Yname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xname=Xname, Yname=Yname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}_{Yname}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}_{Yname}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}_{Yname}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord"])
    #@pytest.mark.parametrize("Yname", ["dihedral_3", "pcoord"])
    @pytest.mark.parametrize("Yname", ["dihedral_3"])
    @pytest.mark.parametrize("Xindex", [0, 1])
    def test_instant_2d_idx(self, Xname, Yname, Xindex):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="instant", Xindex=Xindex,
                                 Xname=Xname, Yname=Yname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"wedap/tests/data/instant_{Xname}{Xindex}_{Yname}_Y.txt"))
        np.testing.assert_allclose(Z, 
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
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/tests/data/average_{Xname}_X.txt"))
        np.testing.assert_allclose(Y, np.loadtxt(f"wedap/tests/data/average_{Xname}_Y.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    @pytest.mark.parametrize("Yname", ["dihedral_3", "dihedral_4"])
    def test_average_2d(self, Xname, Yname):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="average", Xname=Xname, Yname=Yname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}_Z.txt"))
        
    #@pytest.mark.parametrize("Xname", ["dihedral_3", "pcoord"])
    @pytest.mark.parametrize("Xname", ["dihedral_3"])
    @pytest.mark.parametrize("Yname", ["pcoord"])
    @pytest.mark.parametrize("Yindex", [0, 1])
    def test_average_2d_idx(self, Xname, Yname, Yindex):
        X, Y, Z = wedap.H5_Pdist(h5=self.h5, data_type="average", Yindex=Yindex,
                                 Xname=Xname, Yname=Yname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"wedap/tests/data/average_{Xname}_{Yname}{Yindex}_Z.txt"))