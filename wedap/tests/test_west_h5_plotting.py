"""
Unit and regression tests for the west_h5_plotting class.
"""

# Import package, test suite, and other packages as needed
import we_dap
import pytest

import numpy as np
import sys

# look at file coverage for testing
# pytest -v --cov=molecool
# produces .coverage binary file to be used by other tools to visualize 
# do not need 100% coverage, 80-90% is very high

# can have report in 
# $ pytest -v --cov=molecool --cov-report=html
# index.html to better visualize the test coverage

# decorator to skip in pytest
#@pytest.mark.skip

class Test_West_H5_Plotting():
    """
    Test each method of the West_H5_Plotting class.
    """
    # fh_dist = 1                     # distance between 19F-1H (Angstrom)
    # magnet = 1                      # Tesla (600 MHz of 1H+)
    # tc = 1                          # 8.2ns for CypA
    # reduced_anisotropy = 1          # ppm, reduced anisotropy for W4F
    # asymmetry_parameter = 1         # asymmetry parameter for W4F

    # calc_relax = fluorelax.Calc_19F_Relaxation(tc, magnet, fh_dist, reduced_anisotropy, asymmetry_parameter)

    def test_calc_dd_r1(self):
        # calculated_dd_r1 = self.calc_relax.calc_dd_r1()
        # #expected_ff_r1 = 2.0249e-48    # with hbar and gammas
        # expected_dd_r1 = 0.37           # with everything to 1
        # assert pytest.approx(expected_dd_r1) == calculated_dd_r1
