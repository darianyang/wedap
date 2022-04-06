"""
Convert auxillary data recorded during WESTPA simulation and stored in west.h5 file
to various probability density plots.

This script effectively replaces the need to use the native WESTPA plotting pipeline:
west.h5 --w_pdist(with --construct-dataset module.py)--> 
pdist.h5 --plothist(with --postprocess-functions hist_settings.py)--> plot.pdf

TODO: 
    - maybe add option to output pdist as file, this would speed up subsequent plotting
        of the same data.


# Steps:
- first move into plot class and test it out with h5_plot_single
- then seperate into different class methods for evo, instant, average
- then seperate into pdist class and plotting class 
"""

import h5py
import numpy as np
from numpy import inf
import matplotlib.pyplot as plt
from warnings import warn

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

# TODO: add search aux as a class method with seperate args?
class West_H5_Plotting:
    """
    These class methods generate probability distributions and plots the output.
    """

    def __init__(self, h5, data_type, aux_x=None, aux_y=None, first_iter=1, last_iter=None, 
                 bins=100, bin_ext=0.05, p_min=0, p_max=None, p_units='kT'):
        """
        Parameters
        ----------
        h5 : str
            path to west.h5 file
        data_type : str
            'evolution' (1 dataset); 'average' or 'instance' (1 or 2 datasets)
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
        bin_ext : float
            Increase the limits of the bins by a percentage value (0.05 = 5% = default).
        p_min : int
            The minimun probability limit value. Default to 0.
        p_max : int
            The maximum probability limit value.
        p_units : str
            Can be 'kT' (default) or 'kcal'. kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at 298K.
                TODO: make the temp a class attribute or something dynamic.
        # TODO: add data smoothing
        """
        self.h5 = h5
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
        self.bins = bins
        self.bin_ext = bin_ext # TODO: maybe not needed
        self.p_min = p_min # TODO: not set up yet
        self.p_max = p_max
        self.p_units = p_units

    def norm_hist(self, hist,):
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
        if self.p_max: # TODO: this may not be necessary, can just set limits in plotting function
            hist[hist > self.p_max] = inf
            # TODO: westpa makes these the max to keep the pdist shape
        return hist

    def aux_to_pdist(self, iteration, hist_range=None):
        """
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
        f = h5py.File(self.h5, mode="r")
        # each row is walker with 1 column that is a tuple of values, the first being the seg weight
        seg_weights = np.array(f[f"iterations/iter_{iteration:08d}/seg_index"])

        # return 1D aux data: 1D array for histogram and midpoint values
        if self.aux_y == None:
            aux = np.array(f[f"iterations/iter_{iteration:08d}/auxdata/{self.aux_x}"])

            # make an 1-D array to fit the hist values based off of bin count
            histogram = np.zeros(shape=(self.bins))
            for seg in range(0, aux.shape[0]):
                # can use dynamic hist range based off of dataset or a static value from arg
                if hist_range:
                    counts, bins = np.histogram(aux[seg], bins=self.bins, range=hist_range)
                else:
                    counts, bins = np.histogram(aux[seg], bins=self.bins, range=(np.amin(aux), np.amax(aux)))

                # multiply counts vector by weight scalar from seg index 
                counts = np.multiply(counts, seg_weights[seg][0])

                # add all of the weighted walkers to total array for the resulting linear combination
                histogram = np.add(histogram, counts)

            # get bin midpoints
            midpoints_x = (bins[:-1] + bins[1:]) / 2
            
            return midpoints_x, histogram

        # 2D instant histogram and midpoint values for a single specified WE iteration
        if self.aux_y:
            aux_x = np.array(f[f"iterations/iter_{iteration:08d}/auxdata/{self.aux_x}"])
            aux_y = np.array(f[f"iterations/iter_{iteration:08d}/auxdata/{self.aux_y}"])

            # 2D array to store hist counts for each timepoint in both dimensions
            histogram = np.zeros(shape=(self.bins, self.bins))
            for seg in range(0, aux_x.shape[0]):
                # can use dynamic hist range based off of dataset or a static value from arg
                if hist_range:
                    counts, bins_x, bins_y = np.histogram2d(aux_x[seg], aux_y[seg], bins=self.bins, range=hist_range)
                else:
                    counts, bins_x, bins_y = np.histogram2d(aux_x[seg], aux_y[seg], bins=self.bins, 
                                                            range=[[np.amin(aux_x), np.amax(aux_x)], 
                                                                [np.amin(aux_y), np.amax(aux_y)]]
                                                            )

                # multiply counts vector by weight scalar from seg index 
                counts = np.multiply(counts, seg_weights[seg][0])

                # add all of the weighted walkers to total array for the resulting linear combination
                histogram = np.add(histogram, counts)

            # get bin midpoints
            midpoints_x = (bins_x[:-1] + bins_x[1:]) / 2
            midpoints_y = (bins_y[:-1] + bins_y[1:]) / 2
            
            # flip and rotate to correct orientation (alt: TODO, try np.transpose)
            histogram = np.rot90(np.flip(histogram, axis=1))

            return midpoints_x, midpoints_y, histogram


    def get_iter_range(self, aux, iteration):
        """
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
        f = h5py.File(self.h5, mode="r")
        aux_at_iter = np.array(f[f"iterations/iter_{iteration:08d}/auxdata/{aux}"])
        return (np.amin(aux_at_iter) - (np.amin(aux_at_iter) * self.bin_ext * 5), # TODO: this works for now... but need a smarter solution
                np.amax(aux_at_iter) + (np.amax(aux_at_iter) * self.bin_ext)
                )

    def pdist_to_normhist(self):
        """
        Parameters
        ----------

        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using aux_y or evolution (with only aux_x), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        #p_max += 10 # makes for smoother edges of contour plots # TODO, don't really need pmax here anymore
        #p_max = None

        if self.last_iter:
            max_iter = self.last_iter
        elif self.last_iter is None:
            max_iter = h5py.File(self.h5, mode="r").attrs["west_current_iteration"] - 1
        else:
            raise TypeError("last_iter must be int.")

        # get range for max iter hist values: use this as static bin value for evolution plot
        max_iter_hist_range_x = self.get_iter_range(self.aux_x, max_iter)
        if self.aux_y:
            max_iter_hist_range_y = self.get_iter_range(self.aux_y, max_iter)

        if self.data_type == "instant":
            if self.aux_y:
                center_x, center_y, counts_total = self.aux_to_pdist(max_iter, 
                                                                     hist_range=(max_iter_hist_range_x, 
                                                                                 max_iter_hist_range_y
                                                                                 )
                                                                    )
                counts_total = self.norm_hist(counts_total)
                return center_x, center_y, counts_total

            else:
                center, counts_total = self.aux_to_pdist(max_iter, hist_range=max_iter_hist_range_x)
                counts_total = self.norm_hist(counts_total)
                return center, counts_total

        elif self.data_type == "evolution" or self.data_type == "average":
            # make array to store hist (-lnP) values for n iterations of aux_x
            evolution_x = np.zeros(shape=(max_iter, self.bins))
            positions_x = np.zeros(shape=(max_iter, self.bins))
            if self.aux_y:
                average_xy = np.zeros(shape=(self.bins, self.bins))

            for iter in range(self.first_iter, max_iter + 1):
                center_x, counts_total_x = self.aux_to_pdist(iter, hist_range=max_iter_hist_range_x)
                evolution_x[iter - 1] = counts_total_x
                positions_x[iter - 1] = center_x
                
                # 2D avg pdist data generation
                if self.aux_y:
                    center_x, center_y, counts_total_xy = self.aux_to_pdist(iter, hist_range=(max_iter_hist_range_x,  
                                                                                            max_iter_hist_range_y
                                                                                            )
                                                                        )
                    average_xy = np.add(average_xy, counts_total_xy)

            # 2D evolution plot of aux_x (aux_y not used if provided) per iteration        
            if self.data_type == "evolution":
                evolution_x = self.norm_hist(evolution_x)
                return positions_x, np.arange(1, max_iter + 1,1), evolution_x

            if self.data_type == "average":
                # summation of counts for all iterations : then normalize
                col_avg_x = [np.sum(col[col != np.isinf]) for col in evolution_x.T]
                col_avg_x = self.norm_hist(col_avg_x)

                # 2D average plot data for aux_x and aux_y
                if self.aux_y: 
                    average_xy = self.norm_hist(average_xy)
                    return center_x, center_y, average_xy

                # 1D average plot data for aux_x
                else:
                    return center_x, col_avg_x

        else:
            raise ValueError(f"data_type str must be 'instant', 'average', or 'evolution'.\n\tCurrently set to {self.data_type}")

    # TODO
    # def _smooth(self):
    #     if self.data_smoothing_level is not None:
    #         self.Z_data[numpy.isnan(self.Z_data)] = numpy.nanmax(self.Z_data)
    #         self.Z_data = scipy.ndimage.filters.gaussian_filter(self.Z_data, 
    #                             self.data_smoothing_level)
    #     if self.curve_smoothing_level is not None:
    #         self.Z_curves[numpy.isnan(self.Z_curves)] = numpy.nanmax(self.Z_curves)
    #         self.Z_curves = scipy.ndimage.filters.gaussian_filter(self.Z_curves, 
    #                             self.curve_smoothing_level)
    #     self.Z_data[numpy.isnan(self.Z)] = numpy.nan 
    #     self.Z_curves[numpy.isnan(self.Z)] = numpy.nan 

    # TODO: move plotting functions to different file
    def plot_normhist(self, x, y, plot_type="heat", cmap="viridis",  norm_hist=None, ax=None, **plot_options):
        """
        Parameters
        ----------
        x, y : ndarray
            x and y axis values, and if using aux_y or evolution (with only aux_x), also must input norm_hist.
        args_list : argparse.Namespace
            Contains command line arguments passed in by user.
        norm_hist : ndarray
            norm_hist is a 2-D matrix of the normalized histogram values.
        ax : mpl axes object
            args_list options
            -----------------
            plot_type: str
                'heat' (default), or 'contour'. 
            data_type : str
                'evolution' (1 dataset); 'average' or 'instance' (1 or 2 datasets)
            p_max : int
                The maximum probability limit value.
            p_units : str
                Can be 'kT' (default) or 'kcal'. kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at 298K.
            cmap : str
                Colormap option, default = viridis.
            **plot_options : kwargs
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8,6))
        else:
            fig = plt.gcf()

        # 2D heatmaps
        if norm_hist is not None and plot_type == "heat":
            if self.p_max:
                norm_hist[norm_hist > self.p_max] = inf
            plot = ax.pcolormesh(x, y, norm_hist, cmap=cmap, shading="auto", vmin=0, vmax=self.p_max)

        # 2D contour plots TODO: add smooting functionality
        elif norm_hist is not None and plot_type == "contour":
            if self.data_type == "evolution":
                raise ValueError("For contour plot, data_type must be 'average' or 'instant'")
            elif self.p_max is None:
                warn("With 'contour' plot_type, p_max should be set. Otherwise max norm_hist is used.")
                levels = np.arange(0, np.max(norm_hist[norm_hist != np.inf ]), 1)
            elif self.p_max <= 1:
                levels = np.arange(0, self.p_max + 0.1, 0.1)
            else:
                levels = np.arange(0, self.p_max + 1, 1)
            lines = ax.contour(x, y, norm_hist, levels=levels, colors="black", linewidths=1)
            plot = ax.contourf(x, y, norm_hist, levels=levels, cmap=cmap)

        # 1D data
        elif norm_hist is None:
            if self.p_max:
                y[y > self.p_max] = inf
            ax.plot(x, y)
        
        # unpack plot options dictionary # TODO: update this for argparse
        for key, item in plot_options.items():
            if key == "xlabel":
                ax.set_xlabel(item)
            if key == "ylabel":
                ax.set_ylabel(item)
            if key == "xlim":
                ax.set_xlim(item)
            if key == "ylim":
                ax.set_ylim(item)
            if key == "title":
                ax.set_title(item)
            if key == "grid":
                ax.grid(item, alpha=0.5)
            if key == "minima": # TODO: this is essentially bstate, also put maxima?
                # reorient transposed hist matrix
                norm_hist = np.rot90(np.flip(norm_hist, axis=0), k=3)
                # get minima coordinates index (inverse maxima since min = 0)
                maxima = np.where(1 / norm_hist ==  np.amax(1 / norm_hist, axis=(0, 1)))
                # plot point at x and y bin midpoints that correspond to mimima
                ax.plot(x[maxima[0]], y[maxima[1]], 'ko')
                print(f"Minima: ({x[maxima[0]][0]}, {y[maxima[1]][0]})")

        if norm_hist is not None:
            cbar = fig.colorbar(plot)
            # TODO: lines on colorbar?
            #if lines:
            #    cbar.add_lines(lines)
            if self.p_units == "kT":
                cbar.set_label(r"$\Delta F(\vec{x})\,/\,kT$" + "\n" + r"$\left[-\ln\,P(x)\right]$")
            elif self.p_units == "kcal":
                cbar.set_label(r"$\it{-RT}$ ln $\it{P}$ (kcal mol$^{-1}$)")

        #fig.tight_layout()



# TODO: include postprocess.py trace and plot walker functionality

    # TODO: all plotting options with test.h5, compare output
        # 1D Evo, 1D and 2D instant and average
        # optional: diff max_iter and bins args
# TODO: maybe have a yaml config file for plot options
