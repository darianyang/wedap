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

class Test_H5_Pdist():
    """
    Test each method of the H5_Pdist class.
    """
    h5 = "wedap/data/p53.h5"

    # TODO: seperate test for weighted method
    # TODO: add tests for internal pdist methods
    
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    #@pytest.mark.parametrize("weighted", [True, False])
    def test_evolution(self, Xname):
        evolution = wedap.H5_Pdist(self.h5, "evolution", Xname=Xname)
        X, Y, Z = evolution.pdist()

        # X data is the variably filled array of instance pdist x values
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/data/evolution_{Xname}_X.txt"))

        # Y data is just the WE iterations
        np.testing.assert_allclose(Y, 
            np.arange(evolution.first_iter, evolution.last_iter + 1, 1))

        # Z data is the pdist values of each iteration
        np.testing.assert_allclose(Z, np.loadtxt(f"wedap/data/evolution_{Xname}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_instant_1d(self, Xname):
        X, Y, Z = wedap.H5_Pdist(self.h5, "instant", Xname=Xname).pdist()
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/data/instant_{Xname}_X.txt"))
        np.testing.assert_allclose(Y, np.loadtxt(f"wedap/data/instant_{Xname}_Y.txt"))
        
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    @pytest.mark.parametrize("Yname", ["dihedral_3", "dihedral_4"])
    def test_instant_2d(self, Xname, Yname):
        X, Y, Z = wedap.H5_Pdist(self.h5, "instant", Xname=Xname, Yname=Yname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_Z.txt"))
    
    # TODO along with average
    # def test_instant_3d(self):
    #     X, Y, Z = wedap.H5_Pdist(self.h5, "instant", Xname=Xname, Yname=Yname).pdist()
    #     np.testing.assert_allclose(X, 
    #         np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_X.txt"))
    #     np.testing.assert_allclose(Y, 
    #         np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_Y.txt"))
    #     np.testing.assert_allclose(Z, 
    #         np.loadtxt(f"wedap/data/instant_{Xname}_{Yname}_Z.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_average_1d(self, Xname):
        X, Y, Z = wedap.H5_Pdist(self.h5, "average", Xname=Xname).pdist()
        np.testing.assert_allclose(X, np.loadtxt(f"wedap/data/average_{Xname}_X.txt"))
        np.testing.assert_allclose(Y, np.loadtxt(f"wedap/data/average_{Xname}_Y.txt"))

    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    @pytest.mark.parametrize("Yname", ["dihedral_3", "dihedral_4"])
    def test_average_2d(self, Xname, Yname):
        X, Y, Z = wedap.H5_Pdist(self.h5, "average", Xname=Xname, Yname=Yname).pdist()
        np.testing.assert_allclose(X, 
            np.loadtxt(f"wedap/data/average_{Xname}_{Yname}_X.txt"))
        np.testing.assert_allclose(Y, 
            np.loadtxt(f"wedap/data/average_{Xname}_{Yname}_Y.txt"))
        np.testing.assert_allclose(Z, 
            np.loadtxt(f"wedap/data/average_{Xname}_{Yname}_Z.txt"))
