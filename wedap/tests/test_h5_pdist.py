"""
Unit and regression tests for the H5_Pdist class.
"""

from __future__ import unicode_literals

# Import package, test suite, and other packages as needed
import wedap

import numpy as np

from distutils import dir_util
from pytest import fixture
import os

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
    # TODO: need a smaller file (<100Mb) for testing
    h5 = "data/p53.h5"

    # TODO: maybe make one for each function
    # maybe use parameterization to test multiple args like pcoord or other index evo
    def test_evolution(self):
        # using default pcoord 0
        evolution = wedap.H5_Pdist(self.h5, "evolution")
        X, Y, Z = evolution.pdist()
        # X data is the variably filled array of instance pdist x values
        np.testing.assert_array_equal(X, np.loadtxt("tests/evolution_X.txt"))

        # Y data is just the WE iterations
        np.testing.assert_array_equal(Y, 
            np.arange(evolution.first_iter, evolution.last_iter + 1, 1))

        # TODO: need to either seperate this or make a error message list and check empty

data_options = {#"h5" : "data/west_c2.h5",
                #"h5" : "data/multi_2kod.h5",
                "h5" : "data/p53.h5",
                #"Xname" : "1_75_39_c2",
                #"Yname" : "dihedral_4",
                "Xname" : "pcoord",
                "Yname" : "pcoord",
                "Xindex" : 1,
                "Yindex" : 0,            # TODO: maybe can set this more automatically?
                #"Yname" : "angle_3pt",
                #"Xname" : "dihedral_3",
                #"Yname" : "RoG",
                #"Yname" : "XTAL_REF_RMS_Heavy",
                #"Xname" : "rog",
                #"Yname" : "rms_bb_nmr",
                #"Yname" : "rms_bb_xtal",
                #"Yname" : "rms_m1_xtal",
                #"Xname" : "M1_E175_chi2",
                #"Yname" : "M2_E175_chi2",
                "data_type" : "instant",
                #"p_min" : 15,
                #"p_max" : 20,
                #"p_units" : "kcal",
                #"first_iter" : 161,
                #"last_iter" : 161, 
                #"bins" : 100, # note bins affects contour quality
                #"plot_mode" : "contour",
                #"cmap" : "gnuplot_r",
                "plot_mode" : "hist2d",
                #"plot_mode" : "line",
                #"data_smoothing_level" : 0.4,
                #"curve_smoothing_level" : 0.4,
                }

# TODO: eventually use this format to write unit tests of each pdist method

# 2D Example: first initialize the h5 pdist class
#X, Y, Z = H5_Pdist(**data_options).run()
#plt.pcolormesh(X, Y, Z)

#X, Y, Z = H5_Plot(**data_options)
#H5_Plot(X, Y, Z).plot_hist_2d()

# TODO: I should be able to use the classes sepertely or together
#H5_Plot(plot_options=plot_options, **data_options).plot_contour()
#wedap = H5_Pdist(**data_options)
#X, Y, Z = wedap.pdist()

#wedap.plot()
#plt.savefig("west_c2.png")
#print(wedap.auxnames)

# TODO: test using different p_max values and bar

# 1D example
# pdist = H5_Pdist("data/west_i200.h5", X="1_75_39_c2", **data_options)
# X, Y = pdist.run()
# plt.plot(X, Y)
