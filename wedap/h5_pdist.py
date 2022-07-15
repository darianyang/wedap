"""
Convert auxillary data recorded during WESTPA simulation and stored in west.h5 file
to various probability density plots.

This script effectively replaces the need to use the native WESTPA plotting pipeline:
west.h5 --w_pdist(with --construct-dataset module.py)--> 
pdist.h5 --plothist(with --postprocess-functions hist_settings.py)--> plot.pdf

TODO: 
    - maybe add option to output pdist as file, this would speed up subsequent plotting
        of the same data.

TODO: update docstrings

TODO: add option for a list of equivalent h5 files, alternative to w_multi_west.

TODO: have a way to add an arg as an intercepting function to carry out some operation
      on the raw data drom h5, like math or make interval every 10 frames, in _get_data

TODO: working now on only plotting select basis states
      this is done, but need option for output of new h5 file with updated weights
      this will speed up future calculations. In this same line of thought, I should 
      finish the option to output the pdist and use it for plotting instead.
"""

# TEMP for trace plot (TODO)
import matplotlib.pyplot as plt
######

import h5py
import numpy as np
from tqdm.auto import tqdm

# sklearn built on top of scipy but maybe update to only
# have sklearn as dependency (TODO)
#from sklearn.neighbors import KDTree
from scipy.spatial import KDTree

from warnings import warn

# for copying h5 file
import shutil

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

# TODO: maybe can have the plot class take a pdist object as the input
# then if I want to use a loaded pdist, easy to swap it

class H5_Pdist():
    """
    These class methods generate probability distributions from a WESTPA H5 file.
    """
    # TODO: is setting aux_y to None the best approach to 1D plot settings?
    def __init__(self, h5, data_type, Xname="pcoord", Xindex=0, Yname=None, Yindex=0,
                 Zname=None, Zindex=0, first_iter=1, last_iter=None, bins=100, 
                 p_units='kT', T=298, weighted=True, skip_basis=None, skip_basis_out=None,
                 histrange_x=None, histrange_y=None):
        """
        Parameters
        ----------
        h5 : str
            path to west.h5 file
            TODO: list of h5 files.
        data_type : str
            'evolution' (1 dataset); 'average' or 'instant' (1 or 2 datasets)
        Xname : str
            target data for x axis, default pcoord.
        Xindex : int
            If X.ndim > 2, use this to index.
        Yname : str
            target data for y axis, default None.
        Yindex : int
            If Y.ndim > 2, use this to index.
        Zname : str
            target data for z axis, default None. 
            Use this if you want to use a dataset instead of pdist for Z axis.
            This will be best plotted as a scatter plot with Z as the marker color.
            Instead of returning the pdist, only the XYZ datasets will be returned.
            This is becasue the weights/pdist isn't considered.
        Zindex : int
            If Z.ndim > 2, use this to index.
        first_iter : int
            Default start plot at iteration 1 data.
        last_iter : int
            Last iteration data to include, default is the last recorded iteration in the west.h5 file. Note that `instant` type pdists only depend on last_iter.
        bins : int TODO: x and y?
            amount of histogram bins in pdist data to be generated, default 100.
        p_units : str
            Can be 'kT' (default) or 'kcal'. 
            kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at `T` Kelvin.
        T : int
            Temperature if using kcal/mol.
        weighted : bool
            Default True, use WE segment weights in pdist calculation.
        skip_basis : list
            List of binaries for each basis state to determine if it is skipped.
            e.g. [0, 0, 1] would only consider the trajectory data from basis 
            states 1 and 2 but would skip basis state 3, applying zero weights.
        TODO: histrangexy args and docstring, maybe also binsfromexpression?
        """
        # TODO: maybe change self.f to self.h5?
        self.h5 = h5
        self.f = h5py.File(h5, mode="r")
        self.data_type = data_type
        self.p_units = p_units
        self.T = T
        self.weighted = weighted

        # TODO: Default pcoord for either dim
        # add auxdata prefix if not using pcoord
        if Xname != "pcoord":
            Xname = "auxdata/" + Xname
        self.Xname = Xname
        # TODO: set this up as an arg to be able to process 3D+ arrays form aux
        # need to define the index if pcoord is 3D+ array, index is ndim - 1
        self.Xindex = Xindex

        # for 1D plots, but could be better (TODO)
        if Yname is None:
            self.Yname = Yname
        else:
            # add auxdata prefix if not using pcoord
            if Yname != "pcoord":
                Yname = "auxdata/" + Yname
            self.Yname = Yname
            self.Yindex = Yindex

        # for replacing the Z axis pdist with a dataset
        if Zname is None:
            self.Zname = Zname
        else:
            # add auxdata prefix if not using pcoord
            if Zname != "pcoord":
                Zname = "auxdata/" + Zname
        self.Zname = Zname
        self.Zindex = Zindex

        # default to last
        if last_iter is not None:
            self.last_iter = last_iter
        elif last_iter is None:
            self.last_iter = self.f.attrs["west_current_iteration"] - 1

        if data_type == "instant":
            self.first_iter = self.last_iter
        else:
            self.first_iter = first_iter

        # TODO: split into x and y bins
        self.bins = bins

        # save the available aux dataset names]
        # TODO: dosen't work if you don't have an aux dataset dir in h5
        #self.auxnames = list(self.f[f"iterations/iter_{first_iter:08d}/auxdata"])

        # first make a list for each iteration weight array
        weights = []
        #for iter in range(self.first_iter, self.last_iter + 1):
        # have to make array start from iteration 1 to index well during weighting
        # but only for using skipping basis
        if skip_basis is None:
            weight_start = self.first_iter
        elif skip_basis:
            weight_start = 1
        for iter in range(weight_start, self.last_iter + 1):
            weights.append(self.f[f"iterations/iter_{iter:08d}/seg_index"]["weight"])
        # 1D array of variably shaped arrays
        self.weights = np.array(weights, dtype=object)

        self.skip_basis = skip_basis
        self.skip_basis_out = skip_basis_out
        self.histrange_x = histrange_x
        self.histrange_y = histrange_y

    def _get_data_array(self, name, index, iteration):
        """
        Extract, index, and return the aux/data array of interest.
        """
        data = np.array(self.f[f"iterations/iter_{iteration:08d}/{name}"])

        # should work for 1D and 2D pcoords (where 2D is 3D array)
        if data.ndim > 2:
            # get properly indexed dataset
            data = data[:,:,index]

        # add arg for X Y Z and then add Xfun,Yfun,Zfun to init
        # if axis_direction == "X" and self.Xfun (is True):
            # data = Xfun(data)
        return data # TODO: take this and apply the extra function

    # this does add a little overhead at high iteration ranges
    # ~0.5s from 100i to 400i
    # alternatively, can put histrange_x and histrange_y args to skip this
    def _get_histrange(self, name, index):
        """
        Get the histrange considering the min/max of all iterations considered.

        Parameters
        ----------
        aux : str
            target auxillary data for range calculation

        Returns
        -------
        histrange : tuple
            2 item list of min and max bin bounds for hist range of target aux data.
        """
        # set base histrange based on first iteration
        iter_data = self._get_data_array(name, index, self.first_iter)
        histrange = [np.amin(iter_data), np.amax(iter_data)]

        # loop and update to the max and min for all other iterations considered
        for iter in range(self.first_iter + 1, self.last_iter + 1):
            # get min and max for the iteration
            iter_data = self._get_data_array(name, index, iter)
            iter_min = np.amin(iter_data)
            iter_max = np.amax(iter_data)

            # update to get the largest possible range from all iterations
            if iter_min < histrange[0]:
                histrange[0] = iter_min
            if iter_max > histrange[1]:
                histrange[1] = iter_max

        return histrange

    def _normalize(self, hist):
        """
        Parameters
        ----------
        hist : ndarray
            Array containing the histogram count values to be normalized.

        Returns
        -------
        hist : ndarray
            The hist array is normalized according to the p_units argument. 
        """
        # -lnP
        if self.p_units == "kT":
            hist = -np.log(hist / np.max(hist))
        # -RT*lnP
        elif self.p_units == "kcal":
            # Gas constant R = 1.9872 cal/K*mol or 0.0019872 kcal/K*mol
            hist = -0.0019872 * self.T * np.log(hist / np.max(hist))
        else:
            raise ValueError("Invalid p_units value, must be 'kT' or 'kcal'.")
        return hist

    def _get_children_indices(self, parent):
        """
        For a (iter, seg) pair, look for and return all iter + 1 segment indices.

        Parameters
        ----------
        parent : tuple
            (iteration, segment).

        Returns
        -------
        children : tuple
            The indices of all child segments from the input parent segment.
        """
        children = []
        p_iter, p_seg = parent
        # for all parent_id values of the iter + 1 iteration
        for idx, seg in enumerate(
        self.f[f"iterations/iter_{p_iter+1:08d}/seg_index"]["parent_id"]):
            # match with input parent segment
            if seg == p_seg:
                # put all the parent segment children's indices into list
                children.append(idx)
                
        return children

    def _new_weights_from_skip_basis(self):
        """
        Make a new temp h5 file with zero weights for skipped basis state walkers.

        Returns
        -------
        self.weights : array
            Updated weight array with zero values for skipped basis states.
        """
        # setup a warning for h5 files that have incorrectly recorded bstate pcoords
        # this will all be based off of the first pcoord array (Z index 0)
        # correspondingly, the bstate_pcoord will be the first column
        bs_coords = self.f[f"ibstates/0/bstate_pcoord"]
        it1_coords = self.f[f"iterations/iter_00000001/pcoord"][:,0,0]
        # need to second element get the unique indices
        it1_unique_indices = np.unique(it1_coords, return_index=True)[1]
        # then sort to the original bstate ordering
        it1_unique_coords = np.array([it1_coords[index] \
                            for index in sorted(it1_unique_indices)])
        # make sure that traced unique pcoord elements match the basis state values
        if np.isclose(bs_coords[:,0], it1_unique_coords, rtol=1e-04) is False:
            message = f"The traced pcoord \n{it1_unique_coords} \ndoes not equal " + \
                      f"the basis state coordinates \n{bs_coords}"
            warn(message)

        # if the basis state binary is a 1 in skip_basis, use weight 0 
        print("First run skip_basis processing from each initial segment: ")
        for basis, skip in enumerate(self.skip_basis):
            # essentially goes through all initial segments for each skipped basis
            if skip == 1:
                # loop of each initial pcoord value from iteration 1
                for it1_idx, it1_val in enumerate(it1_coords):
                    # so if the pcoord value matches the bstate value to be skipped
                    # needs to both be at the same precision
                    if np.isclose(it1_val, bs_coords[basis,0], rtol=1e-04):
                        # search forward to look for children of basis state 
                        # start at it1_idx, make weight zero 
                        self.weights[0][it1_idx] = 0

                        # list for parent_ids of the current segment skip basis lineage
                        skip_parents_c = [it1_idx]
                        # list for storing the indices to skip for the next iteration
                        skip_parents_n = []

                        # zero the next iteration's children until last_iter
                        for iter in tqdm(range(1, self.last_iter + 1)):
                            for idx in skip_parents_c:
                                # make zero for each child of skip_basis
                                self.weights[iter-1][idx] = 0
                                # then make new skip_parents tuple to loop for next iter
                                skip_parents_n += self._get_children_indices((iter, idx))

                            # make new empty list to store the iteration's skipped
                            skip_parents_c.clear()
                            skip_parents_c += skip_parents_n
                            skip_parents_n.clear()

        # TODO: prob can do better than these print statements
        print("Then run pdist calculation per iteration: ")
        # write new weights into skip_basis_out h5 file
        if self.skip_basis_out is not None:
            shutil.copyfile(self.h5, self.skip_basis_out)
            h5_skip_basis = h5py.File(self.skip_basis_out, "r+")
            for idx, weight in enumerate(self.weights):
                h5_skip_basis[f"iterations/iter_{idx+1:08d}/seg_index"]["weight"] = weight
            
        return self.weights                                

    # TODO: clean up and optimize
    def search_aux_xy_nn(self, val_x, val_y):
        """
        Parameters
        ----------
        # TODO: add step size for searching, right now gets the last frame
        val_x : int or float
        val_y : int or float
        """
        # iter is already known when searching evo data
        if self.data_type == "evolution":
            iter_num = int(val_y)
            
            # These are the auxillary coordinates you're looking for
            r1 = self._get_data_array(self.Xname, self.Xindex, iter_num)[:,-1]

            # phase 2: finding seg number

            # TODO: numpy array this
            small_array2 = []
            for j in range(0,len(r1)):
                small_array2.append([r1[j]])
            tree2 = KDTree(small_array2)

            # TODO: these can be multiple points, maybe can parse these and filter later
            d2, i2 = tree2.query([val_x],k=1)
            seg_num = int(i2)

        else:
            # phase 1: finding iteration number
            distances = []
            indices = []

            # change indices to number of iteration
            for i in range(self.first_iter, self.last_iter + 1): 

                # These are the auxillary coordinates you're looking for
                r1 = self._get_data_array(self.Xname, self.Xindex, i)[:,-1]
                r2 = self._get_data_array(self.Yname, self.Yindex, i)[:,-1]

                small_array = []
                for j in range(0,len(r1)):
                    small_array.append([r1[j],r2[j]])
                tree = KDTree(small_array)

                # Outputs are distance from neighbour (dd) and indices of output (ii)
                dd, ii = tree.query([val_x, val_y],k=1) 
                distances.append(dd) 
                indices.append(ii)

            minimum = np.argmin(distances)
            iter_num = int(minimum+1)

            # These are the auxillary coordinates you're looking for
            r1 = self._get_data_array(self.Xname, self.Xindex, iter_num)[:,-1]
            r2 = self._get_data_array(self.Yname, self.Yindex, iter_num)[:,-1]

            # phase 2: finding seg number

            # TODO: numpy array this
            small_array2 = []
            for j in range(0,len(r1)):
                small_array2.append([r1[j],r2[j]])
            tree2 = KDTree(small_array2)

            # TODO: these can be multiple points, maybe can parse these and filter later
            d2, i2 = tree2.query([val_x, val_y],k=1)
            seg_num = int(i2)

        #print("go to iter " + str(iter_num) + ", " + "and seg " + str(seg_num))
        print(f"Go to ITERATION: {iter_num} and SEGMENT: {seg_num}")
        return iter_num, seg_num

    ##################### TODO: update or organize this #############################
    def get_parents(self, walker_tuple):
        it, wlk = walker_tuple
        parent = self.f[f"iterations/iter_{it:08d}"]["seg_index"]["parent_id"][wlk]
        return it-1, parent

    def trace_walker(self, walker_tuple):
        # Unroll the tuple into iteration/walker 
        it, wlk = walker_tuple
        # Initialize our path
        path = [(it,wlk)]
        # And trace it
        while it > 1: 
            it, wlk = self.get_parents((it, wlk))
            path.append((it,wlk))
        return np.array(sorted(path, key=lambda x: x[0]))

    def get_coords(self, path, data_name, data_index):
        # Initialize a list for the pcoords
        coords = []
        # Loop over the path and get the pcoords for each walker
        for it, wlk in path:
            coords.append(self._get_data_array(data_name, data_index, it)[wlk][::10])
        return np.array(coords)

    def plot_trace(self, walker_tuple, color="white", ax=None):
        """
        Plot trace.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7,5))
        else:
            fig = plt.gcf()

        path = self.trace_walker(walker_tuple)
        # adjustments for plothist evolution of only aux_x data
        if self.data_type == "evolution":
            # split iterations up to provide y-values for each x-value (pcoord)
            aux = self.get_coords(path, self.Xname, self.Xindex)
            iters = np.arange(1, len(aux)+1)
            ax.plot(aux[:,0], iters, c="black", lw=2)
            ax.plot(aux[:,0], iters, c=color, lw=1)
            return

        # And pull aux_coords for the path calculated
        aux_x = self.get_coords(path, self.Xname, self.Xindex)
        aux_y = self.get_coords(path, self.Yname, self.Yindex)

        ax.plot(aux_x[:,0], aux_y[:,0], c="black", lw=2)
        ax.plot(aux_x[:,0], aux_y[:,0], c=color, lw=1)

    ###############################################################################

    def aux_to_pdist_1d(self, iteration):
        """
        Take the auxiliary dataset for a single iteration and generate a weighted
        1D probability distribution. 

        Parameters
        ----------
        iteration : int
            Desired iteration to extract timeseries info from.

        Returns
        -------
        midpoints_x : ndarray
            Histogram midpoint bin values for target aux coordinate of dimension 0.
        midpoints_y : ndarray
            Optional histogram midpoint bin values for target aux coordinate of dimension 1.
        histogram : ndarray
            Raw histogram count values of each histogram bin. Can be later normalized as -lnP(x).
        """
        # return 1D aux data: 1D array for histogram and midpoint values
        aux = self._get_data_array(self.Xname, self.Xindex, iteration)

        # make an 1D array to fit the hist values based off of bin count
        histogram = np.zeros(shape=(self.bins))
        for seg in range(0, aux.shape[0]):
            counts, bins = np.histogram(aux[seg], bins=self.bins, range=self.histrange_x)

            # selectively apply weights
            if self.weighted is True:
                # multiply counts vector by weight scalar from weight array
                counts = np.multiply(counts, self.weights[iteration - self.first_iter][seg])

            # add all of the weighted walkers to total array for the 
            # resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins[:-1] + bins[1:]) / 2
        
        # TODO: also save as instance attributes
        return midpoints_x, histogram

    def aux_to_pdist_2d(self, iteration):
        """
        Take the auxiliary dataset for a single iteration and generate a weighted
        2D probability distribution. 

        Parameters
        ----------
        iteration : int
            Desired iteration to extract timeseries info from.

        Returns
        -------
        midpoints_x : ndarray
            Histogram midpoint bin values for target aux coordinate of dimension 0.
        midpoints_y : ndarray
            Optional histogram midpoint bin values for target aux coordinate of dimension 1.
        histogram : ndarray
            Raw histogram count values of each histogram bin. Can be later normalized as -lnP(x).
        """
        # 2D instant histogram and midpoint values for a single specified WE iteration
        X = self._get_data_array(self.Xname, self.Xindex, iteration)
        Y = self._get_data_array(self.Yname, self.Yindex, iteration)

        # 2D array to store hist counts for each timepoint in both dimensions
        histogram = np.zeros(shape=(self.bins, self.bins))
        for seg in range(0, X.shape[0]):
            counts, bins_x, bins_y = np.histogram2d(X[seg], Y[seg], 
                                                    bins=self.bins, 
                                                    range=[self.histrange_x, 
                                                           self.histrange_y]
                                                    )

            if self.weighted is True:
                # multiply counts vector by weight scalar from weight array
                counts = np.multiply(counts, self.weights[iteration - self.first_iter][seg])

            # add all of the weighted walkers to total array for 
            # the resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins_x[:-1] + bins_x[1:]) / 2
        midpoints_y = (bins_y[:-1] + bins_y[1:]) / 2

        # TODO: save these as instance attributes
        # this will make it easier to save into a text pdist file later

        # save midpoints and transposed histogram (corrected for plotting)
        return midpoints_x, midpoints_y, histogram.T

    def evolution_pdist(self):
        """
        Returns (TODO)
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # make array to store hist (-lnP) values for n iterations of X
        evolution_x = np.zeros(shape=(self.last_iter - self.first_iter + 1, self.bins))
        positions_x = np.zeros(shape=(self.last_iter - self.first_iter + 1, self.bins))

        for iter in tqdm(range(self.first_iter, self.last_iter + 1)):
            # account for first_iter arg for array indexing
            iter_index = iter - self.first_iter + 1
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter)
            evolution_x[iter_index - 1] = counts_total_x
            positions_x[iter_index - 1] = center_x

        # 2D evolution plot of X (Y not used if provided) per iteration        
        evolution_x = self._normalize(evolution_x)

        # bin positions along aux x, WE iteration numbers, z data
        return positions_x, np.arange(self.first_iter, self.last_iter + 1, 1), evolution_x

    # TODO: maybe don't need individual functions, maybe can handle in main
    def instant_pdist_1d(self):
        """
        Returns
        -------
        Xdata, y
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        center, counts_total = self.aux_to_pdist_1d(self.last_iter)
        counts_total = self._normalize(counts_total)
        return center, counts_total

    def instant_pdist_2d(self):
        """
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        center_x, center_y, counts_total = self.aux_to_pdist_2d(self.last_iter)
        counts_total = self._normalize(counts_total)
        return center_x, center_y, counts_total

    def instant_datasets_3d(self):
        """
        Unique case where `Zname` is specified and the XYZ datasets are returned.
        """
        X = self._get_data_array(self.Xname, self.Xindex, self.last_iter)
        # for the case where Zname is specified but not Yname
        if self.Yname is None:
            warn("`Zname` is defined but not `Yname`, using Yname=`pcoord`")
            Y = self._get_data_array("pcoord", self.Yindex, self.last_iter)
        else:
            Y = self._get_data_array(self.Yname, self.Yindex, self.last_iter)
        Z = self._get_data_array(self.Zname, self.Zindex, self.last_iter)

        return X, Y, Z

    def average_pdist_1d(self):
        """
        Returns
        -------
        x, y
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # make array to store hist (-lnP) values for n iterations of X
        evolution_x = np.zeros(shape=(self.last_iter, self.bins))
        positions_x = np.zeros(shape=(self.last_iter, self.bins))

        for iter in tqdm(range(self.first_iter, self.last_iter + 1)):
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter)
            evolution_x[iter - 1] = counts_total_x
            positions_x[iter - 1] = center_x

        # summation of counts for all iterations : then normalize
        col_avg_x = [np.sum(col[col != np.isinf]) for col in evolution_x.T]
        col_avg_x = self._normalize(col_avg_x)

        # 1D average plot data for X
        return center_x, col_avg_x

    def average_pdist_2d(self):
        """
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # empty array for 2D pdist
        average_xy = np.zeros(shape=(self.bins, self.bins))

        # 2D avg pdist data generation
        for iter in tqdm(range(self.first_iter, self.last_iter + 1)):
            center_x, center_y, counts_total_xy = self.aux_to_pdist_2d(iter)
            average_xy = np.add(average_xy, counts_total_xy)

        average_xy = self._normalize(average_xy)
        return center_x, center_y, average_xy

    def average_datasets_3d(self, interval=1):
        """
        Unique case where `Zname` is specified and the XYZ datasets are returned.
        """
        if self.Yname is None:
            warn("`Zname` is defined but not `Yname`, using Yname=`pcoord`")
            self.Yname = "pcoord"

        # get length of each segment
        seg_length = np.array(self.f[f"iterations/iter_{self.first_iter:08d}/pcoord"])
        seg_length = np.shape(seg_length)[1]
    
        # get the total amount of segments in each iteration
        # column 0 of summary is the particles/segments per iteration
        # each row from H5 is a single item that is a tuple of multiple items
        seg_totals = np.array(self.f[f"summary"])
        seg_totals = np.array([i[0] for i in seg_totals])

        # sum only for the iterations considered
        # note that the last_iter attribute is already -1 adjusted
        seg_total = np.sum(seg_totals[self.first_iter - 1:self.last_iter])

        # arrays to be filled with values from each iteration
        # rows are for all segments, columns are each segment datapoint
        X = np.zeros(shape=(seg_total, seg_length))
        Y = np.zeros(shape=(seg_total, seg_length))
        Z = np.zeros(shape=(seg_total, seg_length))

        # loop each iteration
        seg_start = 0
        for iter in tqdm(range(self.first_iter, self.last_iter + 1)):
            # then go through and add all segments/walkers in the iteration
            X[seg_start:seg_start + seg_totals[iter - 1]] = \
                self._get_data_array(self.Xname, self.Xindex, iter)
            Y[seg_start:seg_start + seg_totals[iter - 1]] = \
                self._get_data_array(self.Yname, self.Yindex, iter)
            Z[seg_start:seg_start + seg_totals[iter - 1]] = \
                self._get_data_array(self.Zname, self.Zindex, iter)

            # keeps track of position in the seg_total length based arrays
            seg_start += seg_totals[iter - 1]

        # 3D average datasets using all available data can more managable with interval
        return X[::interval], Y[::interval], Z[::interval]

    def pdist(self, avg3dint=1):
        """
        Main public method with pdist generation controls.

        # TODO: interval may not be needed if I use function arg (Xfun, etc)
        # TODO: maybe make interval for all returns? nah, hist prob doesn't need it?
        """ 
        # option to zero weight out specific basis states
        if self.skip_basis is not None:
            self.weights = self._new_weights_from_skip_basis()

        #print(self.weights)

        # TODO: need to consolidate the Y 2d vs 1d stuff somehow

        # TODO: if I can get rid of this or optimize it, I can then use the 
            # original methods of each pdist by themselves
        # TODO: only if histrange is None
        if self.histrange_x is None:
            # get the optimal histrange
            self.histrange_x = self._get_histrange(self.Xname, self.Xindex)
        # if using 2D pdist
        if self.Yname and self.histrange_y is None:
            self.histrange_y = self._get_histrange(self.Yname, self.Yindex)

        # TODO: need a better way to always return XYZ
        if self.data_type == "evolution":
            return self.evolution_pdist()
        elif self.data_type == "instant":
            if self.Yname and self.Zname:
                return self.instant_datasets_3d()
            elif self.Yname:
                return self.instant_pdist_2d()
            else:
                X, Y = self.instant_pdist_1d()
                return X, Y, np.ones(shape=(self.first_iter, self.last_iter))
        elif self.data_type == "average":
            if self.Yname and self.Zname:
                return self.average_datasets_3d(interval=avg3dint)
            elif self.Yname:
                return self.average_pdist_2d()
            else:
                X, Y = self.average_pdist_1d()
                return X, Y, np.ones(shape=(self.first_iter, self.last_iter))
