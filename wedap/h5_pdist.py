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
TODO: option for unweighted output
"""

import h5py
import numpy as np
from numpy import inf

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

class H5_Pdist:
    """
    These class methods generate probability distributions from a WESTPA H5 file.
    """
    # TODO: is setting aux_y to None the best approach to 1D plot settings?
    def __init__(self, h5, data_type, X="pcoord", Xindex=0, Y=None, Yindex=0,
                 first_iter=1, last_iter=None, bins=100, p_units='kT', T=298):
        """
        Parameters
        ----------
        h5 : str
            path to west.h5 file
        data_type : str
            'evolution' (1 dataset); 'average' or 'instant' (1 or 2 datasets)
        X : str #TODO: default to pcoord1
            target data for x axis
        Y : str #TODO: default to pcoord1
            target data for y axis
        first_iter : int
            Default start plot at iteration 1 data.
        last_iter : int
            Last iteration data to include, default is the last recorded iteration in the west.h5 file.
        bins : int TODO: x and y?
            amount of histogram bins in pdist data to be generated, default 100.
        p_units : str
            Can be 'kT' (default) or 'kcal'. 
            kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at `T` Kelvin.
        T : int
            Temperature if using kcal/mol.
        """
        self.f = h5py.File(h5, mode="r")
        self.data_type = data_type
        self.p_units = p_units
        self.T = T

        # TODO: Default pcoord for either dim
        if X is not "pcoord":
            self.X = "auxdata/" + X
        elif X is "pcoord":
            self.X = "pcoord"
        # TODO: set this up as an arg to be able to process 3D+ arrays form aux
        # need to define the index if pcoord is 3D+ array, index is ndim - 1
        # TODO: can prob do this better
        X = np.array(self.f[f"iterations/iter_{first_iter:08d}/{self.X}"])
        if X.ndim > 2:
            # sets to dim 0 TODO
            self.index_x = np.shape(X)[2] - 2
        else: # TODO: this isn't needed prob
            self.index_x = 0

        # for 1D plots, but could be better (TODO)
        if Y is None:
            self.Y = Y
        else:
            if Y is not "pcoord":
                self.Y = "auxdata/" + Y
            elif Y is "pcoord":
                self.Y = "pcoord"
            Y = np.array(self.f[f"iterations/iter_{first_iter:08d}/{self.Y}"])
            if Y.ndim > 2:
                # sets to dim 1 TODO
                self.index_y = np.shape(Y)[2] - 1
            else:
                self.index_y = 0

        self.first_iter = first_iter
        # default to last
        if last_iter is not None:
            self.last_iter = last_iter
        elif last_iter is None:
            self.last_iter = self.f.attrs["west_current_iteration"] - 1

        # TODO: split into x and y bins
        self.bins = bins

        # save the available aux dataset names
        self.auxnames = list(self.f[f"iterations/iter_{first_iter:08d}/auxdata"])

    def _get_aux_array(self, aux, index, iteration):
        """
        Extract, index, and return the aux array of interest.
        """
        aux_array = np.array(self.f[f"iterations/iter_{iteration:08d}/{aux}"])

        # TODO: should work for 1D and 2D pcoords
        if aux_array.ndim > 2:
            # get properly indexed dataset
            aux_array = aux_array[:,:,index]

        return aux_array

    def _get_histrange(self, aux, index):
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
        iter_aux = self._get_aux_array(aux, index, self.first_iter)
        histrange = [np.amin(iter_aux), np.amax(iter_aux)]

        # loop and update to the max and min for all other iterations considered
        for iter in range(self.first_iter + 1, self.last_iter + 1):
            # get min and max for the iteration
            iter_aux = self._get_aux_array(aux, index, iter)
            iter_min = np.amin(iter_aux)
            iter_max = np.amax(iter_aux)

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
        # the first being the seg weight
        seg_weights = np.array(self.f[f"iterations/iter_{iteration:08d}/seg_index"])

        # return 1D aux data: 1D array for histogram and midpoint values
        aux = self._get_aux_array(self.X, self.index_x, iteration)

        # make an 1-D array to fit the hist values based off of bin count
        histogram = np.zeros(shape=(self.bins))
        for seg in range(0, aux.shape[0]):
            counts, bins = np.histogram(aux[seg], bins=self.bins, range=self.histrange_x)

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
        X = self._get_aux_array(self.X, self.index_x, iteration)
        Y = self._get_aux_array(self.Y, self.index_y, iteration)

        # 2D array to store hist counts for each timepoint in both dimensions
        histogram = np.zeros(shape=(self.bins, self.bins))
        for seg in range(0, X.shape[0]):
            counts, bins_x, bins_y = np.histogram2d(X[seg], Y[seg], 
                                                    bins=self.bins, 
                                                    range=[self.histrange_x, 
                                                           self.histrange_y]
                                                    )

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

    def instant_pdist_1d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        center, counts_total = self.aux_to_pdist_1d(self.last_iter)
        counts_total = self._normalize(counts_total)
        return center, counts_total

    def instant_pdist_2d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        center_x, center_y, counts_total = self.aux_to_pdist_2d(self.last_iter)
        counts_total = self._normalize(counts_total)
        return center_x, center_y, counts_total

    def evolution_pdist(self):
        """ Normalize the Z data
        Returns (TODO)
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # make array to store hist (-lnP) values for n iterations of X
        evolution_x = np.zeros(shape=(self.last_iter, self.bins))
        positions_x = np.zeros(shape=(self.last_iter, self.bins))

        for iter in range(self.first_iter, self.last_iter + 1):
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter)
            evolution_x[iter - 1] = counts_total_x
            positions_x[iter - 1] = center_x

        # 2D evolution plot of X (Y not used if provided) per iteration        
        evolution_x = self._normalize(evolution_x)

        # bin positions along aux x, WE iteration numbers, z data
        return positions_x, np.arange(self.first_iter, self.last_iter + 1, 1), evolution_x

    def average_pdist_1d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # make array to store hist (-lnP) values for n iterations of X
        evolution_x = np.zeros(shape=(self.last_iter, self.bins))
        positions_x = np.zeros(shape=(self.last_iter, self.bins))

        for iter in range(self.first_iter, self.last_iter + 1):
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
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # empty array for 2D pdist
        average_xy = np.zeros(shape=(self.bins, self.bins))

        # 2D avg pdist data generation
        for iter in range(self.first_iter, self.last_iter + 1):
            center_x, center_y, counts_total_xy = self.aux_to_pdist_2d(iter)
            average_xy = np.add(average_xy, counts_total_xy)

        average_xy = self._normalize(average_xy)
        return center_x, center_y, average_xy

    def pdist(self):
        """
        Main public method with pdist generation controls.
        # TODO: put plot controls here? or pdist controls?
                or seperate controls?
        Could add plot methods here and then run the plot methods in each case.
            Or subclass in plot, that is prob best for seperate functionality later on.
            So users can call pdist class or plot class which does both pdist and plot
                or just plots from input data.
        """ 
        # TODO: need to consolidate the Y 2d vs 1d stuff somehow

        # TODO: only if histrange is None
        # get the optimal histrange
        self.histrange_x = self._get_histrange(self.X, self.index_x)
        if self.Y:
            self.histrange_y = self._get_histrange(self.Y, self.index_y)

        if self.data_type == "evolution":
            return self.evolution_pdist()
        elif self.data_type == "instant":
            if self.Y:
                return self.instant_pdist_2d()
            else:
                return self.instant_pdist_1d()
        elif self.data_type == "average":
            if self.Y:
                return self.average_pdist_2d()
            else:
                return self.average_pdist_1d()
