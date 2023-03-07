"""
Unit and regression tests for the H5_Plot class.
"""

# Import package, test suite, and other packages as needed
import wedap

import numpy as np
import pytest

import matplotlib
matplotlib.use('agg')

# look at file coverage for testing
# pytest -v --cov=wedap
# produces .coverage binary file to be used by other tools to visualize 
# do not need 100% coverage, 80-90% is very high

# can have report in 
# $ pytest -v --cov=wedap --cov-report=html
# index.html to better visualize the test coverage

# decorator to skip in pytest
#@pytest.mark.skip

def plot_data_gen(h5, data_type, plot_mode, Xname, Yname=None, Zname=None, out=None):
    """
    Make plot and return or convert to npy binary data file.
    """
    plot = wedap.H5_Plot(h5=h5, data_type=data_type, plot_mode=plot_mode, 
                         Xname=Xname, Yname=Yname, Zname=Zname)
    plot.plot()
    fig = plot.fig

    # draw the figure 
    # tight since canvas is large
    fig.tight_layout(pad=0)
    fig.canvas.draw()

    # Now we can save it to a numpy array.
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # save as np binary file
    if out:
        np.save(out, data)
    
    return data

class Test_H5_Plot():
    """
    Test each method of the H5_Plot class.
    """
    h5 = "wedap/data/p53.h5"
    
    @pytest.mark.parametrize("data_type", ["evolution", "average", "instant"])
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_1_dataset_plots(self, data_type, Xname):
        # make plot data array
        plotdata = plot_data_gen(self.h5, data_type=data_type, plot_mode="line", Xname=Xname)

        # compare to previously generated plot data
        data = np.load(f"wedap/data/plot_{data_type}_line_{Xname}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of 1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500
        
    @pytest.mark.parametrize("data_type", ["average", "instant"])
    @pytest.mark.parametrize("plot_mode", ["hist", "contour"])
    @pytest.mark.parametrize("Xname, Yname", [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]])
    def test_2_dataset_plots(self, data_type, plot_mode, Xname, Yname):
        # make plot data array
        plotdata = plot_data_gen(self.h5, data_type=data_type, plot_mode=plot_mode, 
                                 Xname=Xname, Yname=Yname)

        # compare to previously generated plot data
        data = np.load(f"wedap/data/plot_{data_type}_{plot_mode}_{Xname}_{Yname}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of 1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500
    
    @pytest.mark.parametrize("data_type", ["average", "instant"])
    @pytest.mark.parametrize("Xname, Yname, Zname", [["pcoord", "dihedral_2", "dihedral_3"], 
                                                     ["dihedral_2", "pcoord", "dihedral_3"], 
                                                     ["dihedral_2", "dihedral_3", "pcoord"]])
    def test_3_dataset_plots(self, data_type, Xname, Yname, Zname):
        # make plot data array
        plotdata = plot_data_gen(self.h5, data_type=data_type, plot_mode="scatter3d", 
                                 Xname=Xname, Yname=Yname, Zname=Zname)

        # compare to previously generated plot data
        data = np.load(f"wedap/data/plot_{data_type}_scatter3d_{Xname}_{Yname}_{Zname}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of ~1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500
