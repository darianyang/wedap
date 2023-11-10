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

def plot_data_gen(out=None, show=False, **kwargs):
    """
    Make plot and return or convert to npy binary data file.
    """
    plot = wedap.H5_Plot(**kwargs)
    plot.plot()
    fig = plot.fig

    if show:
        matplotlib.pyplot.show()
        return

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
    
    matplotlib.pyplot.close()

    return data

class Test_H5_Plot():
    """
    Test each method of the H5_Plot class.
    """
    h5 = "wedap/data/p53.h5"
    
    @pytest.mark.parametrize("data_type", ["evolution"])
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_1d_evolution_plots(self, data_type, Xname):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type=data_type, plot_mode="hist", Xname=Xname)

        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_{data_type}_hist_{Xname}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of 1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500

    @pytest.mark.parametrize("data_type", ["average", "instant"])
    @pytest.mark.parametrize("Xname", ["pcoord", "dihedral_2"])
    def test_1d_avg_inst_plots(self, data_type, Xname):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type=data_type, plot_mode="line", Xname=Xname)

        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_{data_type}_line_{Xname}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of 1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500
        

    #@pytest.mark.parametrize("jointplot", [True, False]) # TODO
    @pytest.mark.parametrize("data_type", ["average", "instant"])
    @pytest.mark.parametrize("plot_mode", ["hist", "contour"])
    @pytest.mark.parametrize("Xname, Yname", [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]])
    def test_2_dataset_plots(self, data_type, plot_mode, Xname, Yname):#, jointplot):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type=data_type, plot_mode=plot_mode, 
                                 Xname=Xname, Yname=Yname)#, jointplot=jointplot)

        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_{data_type}_{plot_mode}_{Xname}_{Yname}.npy")
        #data = np.load(f"wedap/tests/data/plot_{data_type}_{plot_mode}_{Xname}_{Yname}_jp{jointplot}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of 1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500
    

    @pytest.mark.parametrize("plot_mode", ["scatter3d", "hexbin3d"])
    @pytest.mark.parametrize("data_type", ["average", "instant"])
    @pytest.mark.parametrize("Xname, Yname, Zname", [["pcoord", "dihedral_2", "dihedral_3"], 
                                                     ["dihedral_2", "pcoord", "dihedral_3"], 
                                                     ["dihedral_2", "dihedral_3", "pcoord"]])
    def test_3_dataset_plots(self, data_type, plot_mode, Xname, Yname, Zname):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type=data_type, plot_mode=plot_mode, 
                                 Xname=Xname, Yname=Yname, Zname=Zname)

        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_{data_type}_{plot_mode}_{Xname}_{Yname}_{Zname}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of ~1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500

    @pytest.mark.parametrize("first_iter, last_iter, step_iter", [[1, 15, 1], 
                                                                  [3, None, 1], 
                                                                  [5, 15, 3]])
    def test_evolution_fi_li_si(self, first_iter, last_iter, step_iter):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type="evolution", plot_mode="hist", 
                                 first_iter=first_iter, last_iter=last_iter, step_iter=step_iter)
        
        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_evolution_hist_fi{first_iter}_li{last_iter}_si{step_iter}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of ~1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500

    def test_evolution_bins_hrx(self, bins=50, hrx=[0, 8]):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type="evolution", plot_mode="hist", 
                                 bins=bins, histrange_x=hrx)
        
        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_evolution_hist_bins{bins}_hrx{hrx[0]}-{hrx[1]}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of ~1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500

    @pytest.mark.parametrize("first_iter, last_iter, step_iter", [[1, 15, 1], 
                                                                  [3, None, 1], 
                                                                  [5, 15, 3]])
    def test_average_fi_li_si(self, first_iter, last_iter, step_iter):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type="average", plot_mode="hist", 
                                 Yname="pcoord", Yindex=1,
                                 first_iter=first_iter, last_iter=last_iter, step_iter=step_iter)
        
        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_average_hist_fi{first_iter}_li{last_iter}_si{step_iter}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of ~1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500

    def test_average_bins_hrx_hry(self, bins=50, hrx=[0, 8], hry=[5, 35]):
        # make plot data array
        plotdata = plot_data_gen(h5=self.h5, data_type="average", plot_mode="hist", 
                                 Yname="pcoord", Yindex=1,
                                 bins=bins, histrange_x=hrx, histrange_y=hry)
        
        # compare to previously generated plot data
        data = np.load(f"wedap/tests/data/plot_average_hist_bins{bins}_hrx{hrx[0]}-{hrx[1]}_hry{hry[0]}-{hry[1]}.npy")
        #np.testing.assert_allclose(plotdata, data)
        # check to see if the amount of mismatches is less than 500 (<1% of ~1 million items)
        assert data.size - np.count_nonzero(plotdata==data) < 500