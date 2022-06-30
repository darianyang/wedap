"""
Python script that uses the KDTree algorithm to search for the nearest neightbor 
to a target 2D coordinate and outputs the segment and iteration numbers.

Adapted from script from JML.
"""

import numpy
import h5py
from scipy.spatial import KDTree

# TODO: eventually put this into main wedap class
# should first reorganize it and essentially re-code it

def find_h5_data(h5, name, iteration, index=0):
    """
    For a iteration, find the dataset by name and properly index it.
    """
    # account for pcoord or aux names
    if name == "pcoord":
        data = h5["iterations"][iteration]["pcoord"]
    else:
        data = h5["iterations"][iteration]["auxdata"][name]

    # account for > 2D arrays such as a 2D+ pcoord
    if data.ndim > 2:
        data = data[:,-1,index]
        #print(data)
        
    else:
        data = data[:,-1]
        #print(data)

    return data

# TODO: update for pcoord (do this better)
# also dosen't work for 2 pcoords, just 1 now
def search_aux_xy_nn(h5, aux_x, aux_y, val_x, val_y, Xindex=0, Yindex=1, last_iter=None, first_iter=1):
    """
    Parameters
    ----------
    # TODO: add step size for searching, right now gets the last frame
    # TODO: add option to search for pcoord and to plot any aux value trace along another aux plot
    h5 : str
        path to west.h5 file
    aux_x : str
        target data for first aux value
    aux_y : str
        target data for second aux value
    val_x : int or float
    val_y : int or float
    last_iter : int
        last iter to consider.
    first_iter : int
        default start at 1.
    """

    f = h5py.File(h5, "r")

    if last_iter:
        max_iter = last_iter
    elif last_iter is None:
        max_iter = f.attrs["west_current_iteration"] - 1

    # phase 1: finding iteration number
    distances = []
    indices = []

    # change indices to number of iteration
    for i in range(first_iter, max_iter + 1): 
        i = str(i)
        iteration = "iter_" + str(numpy.char.zfill(i,8))

        # These are the auxillary coordinates you're looking for
        r1 = find_h5_data(f, aux_x, iteration, index=Xindex)
        r2 = find_h5_data(f, aux_y, iteration, index=Yindex)

        small_array = []
        for j in range(0,len(r1)):
            small_array.append([r1[j],r2[j]])
        tree = KDTree(small_array)

        # Outputs are distance from neighbour (dd) and indices of output (ii)
        dd, ii = tree.query([val_x, val_y],k=1) 
        distances.append(dd) 
        indices.append(ii)

    minimum = numpy.argmin(distances)
    iter_num = int(minimum+1)

    # phase 2: finding seg number
    wheretolook = f"iter_{iter_num:08d}"

    # These are the auxillary coordinates you're looking for
    r1 = find_h5_data(f, aux_x, wheretolook, index=Xindex)
    r2 = find_h5_data(f, aux_y, wheretolook, index=Yindex)

    small_array2 = []
    for j in range(0,len(r1)):
        small_array2.append([r1[j],r2[j]])
    tree2 = KDTree(small_array2)

    # TODO: these can be multiple points, maybe can parse these and filter later
    d2, i2 = tree2.query([val_x, val_y],k=1)
    seg_num = int(i2)

    #print("go to iter " + str(iter_num) + ", " + "and seg " + str(seg_num))
    print(f"Trace plotted for ITERATION: {iter_num} and SEGMENT: {seg_num}")
    return iter_num, seg_num



if __name__ == "__main__": 
    #iter, seg = search_aux_xy_nn("1a43_v02/wcrawl/west_i200_crawled.h5", "1_75_39_c2", "M2Oe_M1He1", 53, 2.8, 200)

    # iter, seg = search_aux_xy_nn("data/west_c2.h5", "1_75_39_c2", "rms_bb_xtal", 80, 6.5, 
    #                             first_iter=1, last_iter=350)

    # iter, seg = search_aux_xy_nn("data/multi_2kod.h5", "1_75_39_c2", "XTAL_REF_RMS_Heavy",
    #                              109, 8.7, first_iter=1, last_iter=200)

    iter, seg = search_aux_xy_nn("data/2kod_v03.02.h5", "1_75_39_c2", "XTAL_REF_RMS_Heavy",
                               103, 8.6, first_iter=1, last_iter=200)