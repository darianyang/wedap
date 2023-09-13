"""
Testing a way to generate a pdist from multiple input h5 files.

The idea being to work in a more convenient way compared to w_multi_west.
So you can just input multiple h5 files and it would weight and plot everything.
Weights are all divided by the n number of input h5 files, so they all still sum to 1.

Features:
    * check shortest file from list (least iters) and use as default last
    * check if requested XYZ dataset exists in all three datasets
        * maybe an option to use multiple dataset names
            * e.g. if data is same but naming differs
    * check that new weights still sum to 1 from multiple h5 files
    * compare to w_multi_west combined h5 file to check
"""

from wedap import H5_Plot, H5_Pdist

import numpy as np
import matplotlib.pyplot as plt

# using 2 of the same files here, so it should equate to the same as 1 file
h5_list = [f"/Users/darian/Desktop/oa_tests/multi-oamax/v0{i}.h5" for i in range(0, 5)]
#h5_list = ["data/p53.h5", "data/p53.h5", "data/p53.h5"]

class H5_Multi(H5_Plot):
    """
    Generate a probability distribution dataset for multiple input h5 files.

    The standard pdist order is; sum all raw probs, then weight, then normalize.

    So for multi: sum each set of raw data values and weight (adjusted for n files),
    then sum all and normalize.
    """

    def __init__(self, h5_list, *args, **kwargs):
        """
        Parameters
        ----------
        h5_list : list of str
            List of h5 file paths.
        """
        self.h5_list = h5_list
        # save if there are requested p_units (TODO: maybe a better way to account this)
        if "p_units" in kwargs:
            self.og_p_units = kwargs["p_units"]
        else:
            self.og_p_units = "kT"
        # have to change to raw for adding counts and will be normalized later
        kwargs["p_units"] = "raw"
        # initialize the parent H5_Pdist class using the first h5 file
        super().__init__(h5=h5_list[0], **kwargs)
        self.kwargs = kwargs

        # TODO: have to go through each file and find a consistent histrange if histrangeXY is None
        # for now, starting by just using the range from the first file (but might not fit all data)
        if self.histrange_x is None:
            # get the optimal histrange
            self.histrange_x = self._get_histrange(self.Xname, self.Xindex)
        # if using 2D pdist
        # needs to handle array input or None input
        if isinstance(self.Yname, (str, np.ndarray)) and self.histrange_y is None:
            self.histrange_y = self._get_histrange(self.Yname, self.Yindex)

    def pdist_scaled(self, h5):
        """
        Instantiate and return a new H5_Pdist object with scaled weights.
        """
        # instantiate the pdist object, have to use raw counts here
        pdist = H5_Pdist(h5, histrange_x=self.histrange_x,
                         histrange_y=self.histrange_y, **self.kwargs)
        # scale the weights based on n h5 files
        pdist.weights /= len(self.h5_list)
        # return the updated pdist object
        return pdist

    def multi_xy(self):
        """
        Returns XYZ pdist arrays for multiple h5 files.
        Normalizes the 1D Y array.
        """
        # make pdist object for first file
        X, Y, Z = self.pdist()

        # loop all other pdists in list and add to initial
        for h5 in self.h5_list[1:]:
            nX, nY, nZ = self.pdist_scaled(h5).pdist()
            Y += nY

        # using p_units back to original values
        # normalize from raw weighted hists
        return X, self._normalize(Y, self.og_p_units), Z

    def multi_xyz(self):
        """
        Returns XYZ pdist arrays for multiple h5 files.
        Normalizes the 2D Z array.
        """
        # make pdist object for first file
        X, Y, Z = self.pdist()

        # loop all other pdists in list and add to initial
        for h5 in self.h5_list[1:]:
            nX, nY, nZ = self.pdist_scaled(h5).pdist()
            Z += nZ

        # using p_units back to original values
        # normalize from raw weighted hists
        return X, Y, self._normalize(Z, self.og_p_units)
    
    def multi_average_datasets_3d(self):
        """
        For scatter3d plots using 3 datasets.
        TODO: add interval
        """
        # make empty arrays shaped n h5_list items * XYZ

        # make pdist object for first file
        X, Y, Z = self.average_datasets_3d()

        # loop all other pdists in list and add to initial
        for h5 in self.h5_list[1:]:
            # create new pdist object and extract 1D data
            nX, nY, nZ = self.average_datasets_3d()
            # TODO: fill out empty array

        return X, Y, Z

    def multi_instant_datasets_3d(self):
        """
        For scatter3d plots using 3 datasets.
        TODO
        """
        pass

    def multi(self):
        """
        Main public class method, returns XYZ pdist arrays for multiple h5 files.
        """
        if self.data_type == "evolution":
            return self.multi_xyz()
        
        elif self.data_type == "instant":
            if self.Yname and self.Zname:
                return self.multi_instant_datasets_3d()
            elif self.Yname:
                # 2D pdist
                return self.multi_xyz()
            else:
                # 1D pdist
                return self.multi_xy()
                
        elif self.data_type == "average":
            # attemts to say, if not None, but has to be compatible with str and arrays
            if isinstance(self.Yname, (str, np.ndarray)) and isinstance(self.Zname, (str, np.ndarray)):
                return self.multi_average_datasets_3d()
            elif isinstance(self.Yname, (str, np.ndarray)):
                # 2D pdist
                return self.multi_xyz()
            else:
                # 1D pdist
                return self.multi_xy()

# run wedap multi test
data_options = {"data_type" : "average",
                "Xname" : "o_angle_m1",
                "Yname" : "o_angle_m2",
                "last_iter" : 10,
                "p_units" : "kT"
                }
# multi = H5_Multi(h5_list, **data_options)
# X, Y, Z = multi.multi()
# multi.plot()
# plt.show()

# TODO: need to potentially subclass H5_plot not H5_pdist (which worked)
# have to make some adjustment to account for this
# then finish adding it to CLI in main and command_line

# could use something like this in the test file (but needs to be smaller file sizes first)
#_, _, checkZ = H5_Pdist("data/p53.h5", "evolution").pdist()
#cX, cY, cZ = H5_Pdist("/Users/darian/Desktop/oa_tests/multi-oamax/wt-v00-04.h5", **data_options).pdist()
#cX, cY, cZ = H5_Pdist("/Users/darian/Desktop/oa_tests/multi-oamax/v00.h5", "evolution", last_iter=10).pdist()
#print(np.testing.assert_array_almost_equal(Z, cZ))
#H5_Plot(cX, cY, cZ).plot()
#plt.show()

pdist = H5_Pdist("/Users/darian/Desktop/oa_tests/multi-oamax/wt-v00-04.h5", **data_options)
print(pdist._get_histrange("pcoord", 0))