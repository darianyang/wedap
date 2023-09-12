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
#h5_list = ["data/p53.h5", "data/p53.h5"]
#h5_list = ["data/p53.h5"]

# NOTE: the order is; sum all raw probs, then weight, then normalize
#       so for multi, maybe try: sum each set of raw and weight, then sum all and normalize
def pdist_new(h5, histrange=[18.744001, 52.316498]):
    # instantiate the pdist object
    pdist = H5_Pdist(h5, data_type="evolution", last_iter=10, p_units="raw", histrange_x=histrange)
    # scale the weights based on n h5 files
    pdist.weights /= len(h5_list)
    # calculate updated pdist
    return pdist

def multi_h5(h5_list):
    # make inital pdist
    i_pdist = pdist_new(h5_list[0])
    X, Y, Z = i_pdist.pdist()

    # set inf to 0 (so we can add and divide probabilities)
    #Z[Z == np.inf] = 0 

    # loop all other pdists in list and add to initial
    for h5 in h5_list[1:]:
        nX, nY, nZ = pdist_new(h5).pdist()
        Z += nZ

    i_pdist.p_units = "kT"
    # normalize from raw weighted hists
    return X, Y, i_pdist._normalize(Z)

# run wedap multi test
X, Y, Z = multi_h5(h5_list)

#_, _, checkZ = H5_Pdist("data/p53.h5", "evolution").pdist()
#cX, cY, cZ = H5_Pdist("/Users/darian/Desktop/oa_tests/multi-oamax/wt-v00-04.h5", "evolution", last_iter=10).pdist()
#cX, cY, cZ = H5_Pdist("/Users/darian/Desktop/oa_tests/multi-oamax/v00.h5", "evolution", last_iter=10).pdist()
#print(np.testing.assert_array_almost_equal(Z, cZ))

H5_Plot(X, Y, Z).plot()
#H5_Plot(cX, cY, cZ).plot()
plt.show()

# if __name__ == "__main__":
#     pdist = H5_Pdist("/Users/darian/Desktop/oa_tests/multi-oamax/wt-v00-04.h5", "evolution", last_iter=10)
#     histrange = pdist._get_histrange("pcoord", 0)
#     print(histrange)