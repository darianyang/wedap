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
"""

import h5py
import numpy as np
from numpy import inf
from tqdm.auto import tqdm

from warnings import warn

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

class H5_Pdist():
    """
    These class methods generate probability distributions from a WESTPA H5 file.
    """
    # TODO: is setting aux_y to None the best approach to 1D plot settings?
    def __init__(self, h5, data_type, Xname="pcoord", Xindex=0, Yname=None, Yindex=0,
                 Zname=None, Zindex=0, first_iter=1, last_iter=None, bins=100, 
                 p_units='kT', T=298, weighted=True):
        """
        Parameters
        ----------
        h5 : str
            path to west.h5 file
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
        TODO: histrangexy args, maybe also binsfromexpression?
        """
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

        # save the available aux dataset names
        self.auxnames = list(self.f[f"iterations/iter_{first_iter:08d}/auxdata"])

    def _get_data_array(self, name, index, iteration):
        """
        Extract, index, and return the aux/data array of interest.
        """
        data = np.array(self.f[f"iterations/iter_{iteration:08d}/{name}"])

        # TODO: should work for 1D and 2D pcoords
        if data.ndim > 2:
            # get properly indexed dataset
            data = data[:,:,index]

        return data

    # TODO: this does add a little overhead at high iteration ranges
        # ~0.5s from 100i to 400i
        # maybe can find a more efficient way
        # before I just took the max of the last iteration but this lead to some 
            # issues with not catching the entire dist (hence bin_ext)
            # how does w_pdist/plothist do this? how about AJD?
        # need to add option for custom, then make limits plug into histranges?
    def _get_histrange(self, name, index):
        """
        Get the histrange considering the min/max of all iterations considered.
        TODO: this is currently done before the pdist gen, but maybe better during
            the pdist gen loop for efficiency, but problem is, need histrange for pdist.

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
        # each row is walker with 1 column that is a tuple of values
        # the first being the seg weight (TODO: weighting internal method?)
        seg_weights = np.array(self.f[f"iterations/iter_{iteration:08d}/seg_index"])

        # return 1D aux data: 1D array for histogram and midpoint values
        aux = self._get_data_array(self.Xname, self.Xindex, iteration)

        # make an 1-D array to fit the hist values based off of bin count
        histogram = np.zeros(shape=(self.bins))
        for seg in range(0, aux.shape[0]):
            counts, bins = np.histogram(aux[seg], bins=self.bins, range=self.histrange_x)

            # TODO: these parts would also be in the weighting internal method
            if self.weighted is True:
                # multiply counts vector by weight scalar from seg index 
                counts = np.multiply(counts, seg_weights[seg][0])

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
        # each row is walker with 1 column that is a tuple of values
        # the first being the seg weight
        seg_weights = np.array(self.f[f"iterations/iter_{iteration:08d}/seg_index"])

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
                # multiply counts vector by weight scalar from seg index 
                counts = np.multiply(counts, seg_weights[seg][0])

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
        evolution_x = np.zeros(shape=(self.last_iter, self.bins))
        positions_x = np.zeros(shape=(self.last_iter, self.bins))

        for iter in tqdm(range(self.first_iter, self.last_iter + 1)):
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter)
            evolution_x[iter - 1] = counts_total_x
            positions_x[iter - 1] = center_x

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

    def average_datasets_3d(self, interval=10):
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

    def pdist(self, avg3dint=10):
        """
        Main public method with pdist generation controls.

        # TODO: maybe make interval for all returns? nah, hist prob doesn't need it?
        """ 
        # TODO: need to consolidate the Y 2d vs 1d stuff somehow

        # TODO: only if histrange is None
        # TODO: if I can get rid of this or optimize it, I can then use the 
            # original methods of each pdist by themselves
        # get the optimal histrange
        self.histrange_x = self._get_histrange(self.Xname, self.Xindex)
        if self.Yname:
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
