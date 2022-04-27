"""
Convert auxillary data recorded during WESTPA simulation and stored in west.h5 file
to various probability density plots.

This script effectively replaces the need to use the native WESTPA plotting pipeline:
west.h5 --w_pdist(with --construct-dataset module.py)--> 
pdist.h5 --plothist(with --postprocess-functions hist_settings.py)--> plot.pdf

TODO: 
    - maybe add option to output pdist as file, this would speed up subsequent plotting
        of the same data.

# TODO: add Z-bins option? or just add contour level option to plotting class
    # this is equivalent to the Z bins

# Steps:
- first move into plot class and test it out with h5_plot_single (done)
- then seperate into different class methods for evo, instant, average
- then seperate into pdist class and plotting class 

TODO: update docstrings
"""

import h5py
from matplotlib.pyplot import hist
import numpy as np
from numpy import inf

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

class H5_Pdist:
    """
    These class methods generate probability distributions and plots the output.
    TODO: split?
    """
    # TODO: change aux_x to X?
    def __init__(self, h5, data_type, aux_x=None, aux_y=None, first_iter=1, last_iter=None, 
                 bins=100, bin_ext=0.1, p_min=0, p_max=None, p_units='kT'):
        """
        Parameters
        ----------
        h5 : str
            path to west.h5 file
        data_type : str
            'evolution' (1 dataset); 'average' or 'instant' (1 or 2 datasets)
        aux_x : str #TODO: default to pcoord1
            target data for x axis
        aux_y : str #TODO: default to pcoord1
            target data for y axis
        first_iter : int
            Default start plot at iteration 1 data.
        last_iter : int
            Last iteration data to include, default is the last recorded iteration in the west.h5 file.
        bins : int
            amount of histogram bins in pdist data to be generated, default 100.
        bin_ext : float TODO
            Increase the limits of the bins by a percentage value (0.05 = 5% = default).
        p_min : int
            The minimun probability limit value. Default to 0.
        p_max : int
            The maximum probability limit value.
        p_units : str
            Can be 'kT' (default) or 'kcal'. kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at 298K.
                TODO: make the temp a class attribute or something dynamic.
        TODO: arg for histrange_x and histrange_y, can use xlim and ylim if provided in H5_Plot
        """
        self.f = h5py.File(h5, mode="r")

        self.data_type = data_type

        # TODO: make these instance attributes equal the h5 target directory, but default pcoord
        # Default pcoord for either dim
        if aux_x is not None:
            #self.aux_x = np.array(f[f"iterations/iter_{iteration:08d}/auxdata/{aux_x}"]) # TODO: maybe just the last part of string
            self.aux_x = aux_x
        elif aux_x is None:
            #self.aux_x = np.array(f[f"pcoord0"])
            self.aux_x = aux_x

        self.aux_y = aux_y

        self.first_iter = first_iter
        # default to last if not None
        if last_iter is not None:
            self.last_iter = last_iter
        elif last_iter is None:
            self.last_iter = h5py.File(h5, mode="r").attrs["west_current_iteration"] - 1

        # TODO: split into x and y bins
        self.bins = bins
        self.bin_ext = bin_ext # TODO: maybe not needed
        self.p_min = p_min # TODO: not set up yet
        self.p_max = p_max
        self.p_units = p_units

        # TODO: this needs to be updated for periodic values
            # current workaround for periodic torsion data:
                # use bin_ext of <= 0.01
        # hist_range # TODO: this will supercede bin_ext
        # can take the largest range on both dims for the histrange of evo and average
            # instant will be just the single dist range

    def _get_hist_range(self, aux):
        """ 
        Get the proper instance attribute considering the min/max of the entire dataset.

        Parameters
        ----------
        aux : str
            target auxillary data for range calculation

        Returns
        -------
        histrange : tuple
            2 item tuple of min and max bin bounds for hist range of target aux data.
        """
        # original min and max histrange values
        histrange = [0,0]

        # loop and update to the max and min for all iterations considered
        for iter in range(self.first_iter, self.last_iter + 1):
            aux = np.array(self.f[f"iterations/iter_{iter:08d}/auxdata/{aux}"])
            
            # update to get the largest possible range for all iterations
            if np.amin(aux) < histrange[0]:
                histrange[0] = np.amin(aux)
            if np.amax(aux) > histrange[1]:
                histrange[1] = np.amax(aux)

        return histrange


    def get_iter_range(self, aux, iteration):
        """ TODO: make internal method?
        Parameters
        ----------
        aux : str
            target auxillary data for range calculation
        iteration : int
            iteration to calculate range of

        Returns
        -------
        iter_range : tuple
            2 item tuple of min and max bin bounds for hist range of target aux data.
        """
        aux_at_iter = np.array(self.f[f"iterations/iter_{iteration:08d}/auxdata/{aux}"])
        # TODO: this *5 works for now... but need a smarter solution
        return (np.amin(aux_at_iter) - (np.amin(aux_at_iter) * self.bin_ext), 
                np.amax(aux_at_iter) + (np.amax(aux_at_iter) * self.bin_ext)
                )

    def _normalize(self, hist):
        """ TODO: add temperature arg, also this function may not be needed.
        Parameters
        ----------
        hist : ndarray
            Array containing the histogram count values to be normalized.

        Returns
        -------
        hist : ndarray
            The hist array is normalized according to the p_units argument. 
            If p_max, probability values above p_max are adjusted to be inf.
        """
        if self.p_units == "kT":
            hist = -np.log(hist / np.max(hist))
        elif self.p_units == "kcal":
            hist = -0.5922 * np.log(hist / np.max(hist))
        else:
            raise ValueError("Invalid p_units value, must be 'kT' or 'kcal'.")
        if self.p_max: # TODO: this may not be necessary, can just set limits in plotting function, or maybe it would be best to make the final pdist here and just leave plotting to the plot class
            hist[hist > self.p_max] = inf
            # TODO: westpa makes these the max to keep the pdist shape
        return hist

    def aux_to_pdist_1d(self, iteration, hist_range=None):
        """
        Take the auxiliary dataset for a single iteration and generate a weighted
        1D probability distribution. 

        Parameters
        ----------
        iteration : int
            Desired iteration to extract timeseries info from.
        hist_range: tuple (optional)
            2 int values for min and max hist range.

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
        aux = np.array(self.f[f"iterations/iter_{iteration:08d}/auxdata/{self.aux_x}"])

        # make an 1-D array to fit the hist values based off of bin count
        histogram = np.zeros(shape=(self.bins))
        for seg in range(0, aux.shape[0]):
            # can use dynamic hist range based off of dataset or a static value from arg
            # TODO: i dont think this option is ever used
            if hist_range is None:
                hist_range = (np.amin(aux), np.amax(aux))
            counts, bins = np.histogram(aux[seg], bins=self.bins, range=hist_range)

            # multiply counts vector by weight scalar from seg index 
            counts = np.multiply(counts, seg_weights[seg][0])

            # add all of the weighted walkers to total array for the 
            # resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins[:-1] + bins[1:]) / 2
        
        # TODO: save as instance attributes
        return midpoints_x, histogram

    def aux_to_pdist_2d(self, iteration, hist_range=None):
        """
        Take the auxiliary dataset for a single iteration and generate a weighted
        2D probability distribution. 

        Parameters
        ----------
        iteration : int
            Desired iteration to extract timeseries info from.
        hist_range: tuple (optional)
            2 int values for min and max hist range.

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
        aux_x = np.array(self.f[f"iterations/iter_{iteration:08d}/auxdata/{self.aux_x}"])
        aux_y = np.array(self.f[f"iterations/iter_{iteration:08d}/auxdata/{self.aux_y}"])

        # 2D array to store hist counts for each timepoint in both dimensions
        histogram = np.zeros(shape=(self.bins, self.bins))
        for seg in range(0, aux_x.shape[0]):
            # can use dynamic hist range based off of dataset or a static value from arg
            if hist_range is None:
                hist_range = [[np.amin(aux_x), np.amax(aux_x)], 
                              [np.amin(aux_y), np.amax(aux_y)]]
            counts, bins_x, bins_y = np.histogram2d(aux_x[seg], aux_y[seg], 
                                                    bins=self.bins, range=hist_range)

            # multiply counts vector by weight scalar from seg index 
            counts = np.multiply(counts, seg_weights[seg][0])

            # add all of the weighted walkers to total array for 
            # the resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins_x[:-1] + bins_x[1:]) / 2
        midpoints_y = (bins_y[:-1] + bins_y[1:]) / 2
        
        # flip and rotate to correct orientation (alt: TODO, try np.transpose)
        histogram = np.rot90(np.flip(histogram, axis=1))

        # TODO: save these as instance attributes
        # this will make it easier to save into a text pdist file later
        return midpoints_x, midpoints_y, histogram

    def instant_pdist_1d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using aux_y or evolution (with only aux_x), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # get range for max iter hist values: use this as static bin value for evolution plot
        hist_range_x = self.get_iter_range(self.aux_x, self.last_iter)
        center, counts_total = self.aux_to_pdist_1d(self.last_iter, hist_range=hist_range_x)
        counts_total = self._normalize(counts_total)
        return center, counts_total

    def instant_pdist_2d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using aux_y or evolution (with only aux_x), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # TODO: put histrange x and y in init
        # get range for max iter hist values: use this as static bin value for evolution plot
        hist_range_x = self.get_iter_range(self.aux_x, self.last_iter)
        hist_range_y = self.get_iter_range(self.aux_y, self.last_iter)
        center_x, center_y, counts_total = self.aux_to_pdist_2d(self.last_iter, 
                                                                hist_range=(hist_range_x,
                                                                            hist_range_y
                                                                            )
                                                                )
        counts_total = self._normalize(counts_total)
        return center_x, center_y, counts_total

    def evolution_pdist(self):
        """ Nor malize the Z data
        Returns (TODO)
        -------
        x, y, norm_hist
            x and y axis values, and if using aux_y or evolution (with only aux_x), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # get range for max iter hist values: use this as static bin value for evolution plot
        hist_range_x = self.get_iter_range(self.aux_x, self.last_iter)

        # make array to store hist (-lnP) values for n iterations of aux_x
        evolution_x = np.zeros(shape=(self.last_iter, self.bins))
        positions_x = np.zeros(shape=(self.last_iter, self.bins))

        for iter in range(self.first_iter, self.last_iter + 1):
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter, hist_range=hist_range_x)
            evolution_x[iter - 1] = counts_total_x
            positions_x[iter - 1] = center_x

        # 2D evolution plot of aux_x (aux_y not used if provided) per iteration        
        evolution_x = self._normalize(evolution_x)

        # bin positions along aux x, WE iteration numbers, z data
        return positions_x, np.arange(self.first_iter, self.last_iter + 1,1), evolution_x

    def average_pdist_1d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using aux_y or evolution (with only aux_x), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # get range for max iter hist values: use this as static bin value for evolution plot
        hist_range_x = self.get_iter_range(self.aux_x, self.last_iter)

        # make array to store hist (-lnP) values for n iterations of aux_x
        evolution_x = np.zeros(shape=(self.last_iter, self.bins))
        positions_x = np.zeros(shape=(self.last_iter, self.bins))

        for iter in range(self.first_iter, self.last_iter + 1):
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter, hist_range=hist_range_x)
            evolution_x[iter - 1] = counts_total_x
            positions_x[iter - 1] = center_x

        # summation of counts for all iterations : then normalize
        col_avg_x = [np.sum(col[col != np.isinf]) for col in evolution_x.T]
        col_avg_x = self._normalize(col_avg_x)

        # 1D average plot data for aux_x
        return center_x, col_avg_x

    def average_pdist_2d(self):
        """ Normalize the Z data
        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using aux_y or evolution (with only aux_x), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # get range for max iter hist values: use this as static bin value for evolution plot
        hist_range_x = self.get_iter_range(self.aux_x, self.last_iter)
        hist_range_y = self.get_iter_range(self.aux_y, self.last_iter)

        # empty array for 2D pdist
        average_xy = np.zeros(shape=(self.bins, self.bins))

        # 2D avg pdist data generation
        for iter in range(self.first_iter, self.last_iter + 1):
            center_x, center_y, counts_total_xy = \
                self.aux_to_pdist_2d(iter, hist_range=(hist_range_x, hist_range_y))
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
        if self.data_type == "evolution":
            return self.evolution_pdist()
        elif self.data_type == "instant":
            if self.aux_y:
                return self.instant_pdist_2d()
            else:
                return self.instant_pdist_1d()
        elif self.data_type == "average":
            if self.aux_y:
                return self.average_pdist_2d()
            else:
                return self.average_pdist_1d()
