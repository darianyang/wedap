"""
Main plotting class of wedap.
Plot all of the datasets generated with H5_Pdist.

# TODO: all plotting options with test.h5, compare output
    # 1D Evo, 1D and 2D instant and average
    # optional: diff max_iter and bins args

TODO: maybe make methods for the following plots:
contourf--plot contour levels
histogram--plot histogram.
lines--plot contour lines only.
contourf_l--plot contour levels and lines.
histogram_l--plot histogram and contour lines.
option - with and without side histograms
- mpl mosaic options
- see mpl scatter hist: https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html
maybe a ridgeline plot?
- This would be maybe for 1D avg of every 100 iterations
- https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
Option to overlay different datasets, could be done easily with python but maybe a cli option?                

TODO: plot clustering centroids option?
      can then grab the search_aux at the centroid

TODO: bin visualizer? and maybe show the trajectories as just dots?
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from warnings import warn
from numpy import inf

from .h5_pdist import *

# TODO: maybe put the pdist object into the plot class and have this object be flexible
# so it could just be a pdist.h5 file from westpa or make your own
# or read in a pdist.h5 and create pdist object using that dataset

class H5_Plot(H5_Pdist):
    """
    These methods provide various plotting options for pdist data.
    """
    def __init__(self, X=None, Y=None, Z=None, plot_mode="hist", cmap=None, smoothing_level=None,
        color=None, ax=None, p_min=None, p_max=None, contour_interval=1, contour_levels=None,
        cbar_label=None, cax=None, jointplot=False, *args, **kwargs):
        """
        Plotting of pdists generated from H5 datasets.

        Parameters
        ----------
        X, Y : arrays
            x and y axis values, and if using aux_y or evolution (with only aux_x), also must input Z.
        Z : 2darray
            Z is a 2-D matrix of the normalized histogram values.
        plot_mode : str
            TODO: update and expand. Can be 'hist' (default), 'contour', 'line', 'scatter3d'.
        cmap : str
            Can be string or cmap to be input into mpl. Default = viridis.
        smoothing_level : float
            Optionally add gaussian noise to smooth Z data. A good value is around 0.4 to 1.0.
        color : str
            Color for 1D plots.
        ax : mpl axes object
        plot_options : kwargs dictionary
            Include mpl based plot options (e.g. xlabel, ylabel, ylim, xlim, title).
        p_min : int
            The minimum probability limit value.
        p_max : int
            The maximum probability limit value.
        contour_interval : int
            Interval to put contour levels if using 'contour' plot_mode.
        cbar_label : str
            Label for the colorbar.
        cax : MPL axes object
            Optionally define axes object to place colorbar.
        jointplot : bool
            Whether or not to include marginal plots. Note to use this argument, 
            probabilities for Z or from H5_Pdist must be in `raw` p_units.
        ** args
        ** kwargs
        """
        # include the init args for H5_Pdist
        # TODO: how to make some of the args optional if I want to use classes seperately?
        #super().__init__(*args, **kwargs)

        self.ax = ax
        self.smoothing_level = smoothing_level
        self.jointplot = jointplot

        # TODO: option if you want to generate pdist
        # also need option of just using the input X Y Z args
        # or getting them from w_pdist h5 file, or from H5_Pdist output file
        # user inputs XYZ
        if X is None and Y is None and Z is None:
            super().__init__(*args, **kwargs)
            # save the user requested p_units and changes p_units to raw
            if self.jointplot:
                self.requested_p_units = self.p_units
                kwargs["p_units"] = "raw"
            # will be re-normed later on
            X, Y, Z = H5_Pdist(*args, **kwargs).pdist()

        self.X = X
        self.Y = Y
        self.Z = Z

        self.p_min = p_min
        self.p_max = p_max
        self.contour_interval = contour_interval
        self.contour_levels = contour_levels

        self.plot_mode = plot_mode
        self.cmap = cmap
        self.color = color # 1D color
        #self.plot_options = plot_options

        # set cbar_label to default to blank if None
        if cbar_label is None:
            self.cbar_label = ""

        # if no p_units are there then no label is fine
        # otherwise check if p_units are there
        if hasattr(self, "p_units"):
            if self.p_units == "kT":
                self.cbar_label = "$-\ln\,P(x)$"
            elif self.p_units == "kcal":
                self.cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"
            elif self.p_units == "raw":
                self.cbar_label = "Counts"
            elif self.p_units == "raw_norm":
                self.cbar_label = "Normalized Counts"
        # if using 3 datasets, put blank name as default cbar
        if self.plot_mode == "scatter3d":
            self.cbar_label = ""
        # overwrite and apply cbar_label attr if available/specified
        if cbar_label:
            self.cbar_label = cbar_label

        self.cax = cax
        self.kwargs = kwargs

    # TODO: load from w_pdist, also can add method to load from wedap pdist output
    # def _load_from_pdist_file(self):
    #     '''
    #     Load data from a w_pdist output file. This includes bin boundaries. 
    #     '''
    #     # Open the HDF5 file.
    #     self.pdist_HDF5 = h5py.File(self.args.pdist_file)

    #     # Load the histograms and sum along all axes except those specified by
    #     # the user.  Also, only include the iterations specified by the user.
    #     histogram      = numpy.array(self.pdist_HDF5['histograms'])

    #     # Figure out what iterations to use
    #     n_iter_array   = numpy.array(self.pdist_HDF5['n_iter'])
    #     if self.args.first_iter is not None:
    #         first_iter = self.args.first_iter
    #     else:
    #         first_iter = n_iter_array[0] 
    #     if self.args.last_iter is not None:
    #         last_iter = self.args.last_iter
    #     else:
    #         last_iter = n_iter_array[-1]
    #     first_iter_idx = numpy.where(n_iter_array == first_iter)[0][0]
    #     last_iter_idx  = numpy.where(n_iter_array == last_iter)[0][0]
    #     histogram      = histogram[first_iter_idx:last_iter_idx+1]

    #     # Sum along axes
    #     self.axis_list = self._get_bins_from_expr(self.args.pdist_axes)
    #     self.H         = self._sum_except_along(histogram, self.axis_list) 

    #     # Make sure that the axis ordering is correct.
    #     if self.axis_list[0] > self.axis_list[1]:
    #         self.H = self.H.transpose()

    def add_cbar(self, cax=None):
        """
        Add cbar.

        Parameters
        ----------
        cax : mpl cbar axis
            Optionally specify the cbar axis.
        """
        # fig vs plt should be the same, tests run fine (needed to go plt for mosaic)
        #cbar = self.fig.colorbar(self.plot, cax=cax)
        cbar = plt.colorbar(self.plot_obj, cax=cax)
        # if contour lines are present
        if hasattr(self, "lines"):
            cbar.add_lines(self.lines)

        # TODO: move labelpad here to style?
        cbar.set_label(self.cbar_label, labelpad=14)

        # allow for cbar object manipulation (e.g. removal in movie)
        self.cbar = cbar
    
    def plot_hist(self):
        """
        2d hist plot.
        """
        # 2D heatmaps
        # TODO: westpa makes these the max to keep the pdist shape
        # if self.p_max:
        #     self.Z[self.Z > self.p_max] = inf
        self.plot_obj = self.ax.pcolormesh(self.X, self.Y, self.Z, cmap=self.cmap, 
                                           shading="auto", vmin=self.p_min, vmax=self.p_max)

    def _get_contour_levels(self):
        """
        Get contour level attribute.
        """
        # TODO: could clean up this logic better
        if self.p_min is None:
            self.p_min = np.min(self.Z)
        # if levels aren't specified
        if self.contour_levels is None:
            # 2D contour plots
            if self.p_max is None:
                warn("With 'contour' plot_type, p_max should be set. Otherwise max Z is used.")
                self.contour_levels = np.arange(self.p_min, np.max(self.Z[self.Z != np.inf ]), self.contour_interval)
            elif self.p_max <= 1:
                warn("You may want to change the `contour_interval` argument to be < 1")
                self.contour_levels = np.arange(self.p_min, self.p_max + self.contour_interval, self.contour_interval)
            else:
                self.contour_levels = np.arange(self.p_min, self.p_max + self.contour_interval, self.contour_interval)

    def plot_contour_l(self):
        """
        2d contour plot, lines.
        """
        # can control linewidths using rc params (lines.linewidths (default 1.5))
        if self.color:
            self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=self.contour_levels, colors=self.color)
            #self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=[5], colors=self.color)
        else:
            self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=self.contour_levels, cmap=self.cmap)

    def plot_contour_f(self):
        """
        2d contour plot, fill.
        """
        self.plot_obj = self.ax.contourf(self.X, self.Y, self.Z, levels=self.contour_levels, cmap=self.cmap)

    def plot_bar(self):
        """
        Simple bar plot.
        """
        # 1D data
        self.ax.bar(self.X, self.Y, color=self.color)
        self.ax.set_ylabel("P(x)")

    def plot_line(self):
        """
        1d line plot.
        """
        # 1D data
        if self.p_max:
            self.Y[self.Y > self.p_max] = inf
        self.ax.plot(self.X, self.Y, color=self.color)
        self.ax.set_ylabel(self.cbar_label)
    
    def plot_scatter3d(self, interval=10, s=1):
        """
        3d scatter plot.

        Parameters
        ----------
        interval : int
            Interval to consider the XYZ datasets, increase to use less data.
        s : float
            mpl scatter marker size.
        """
        self.plot_obj = self.ax.scatter(self.X[::interval], 
                                        self.Y[::interval], 
                                        c=self.Z[::interval], 
                                        cmap=self.cmap, s=s,
                                        vmin=self.p_min, vmax=self.p_max)

    def plot_hexbin3d(self):
        """
        Hexbin plot?
        """
        # TODO: test this and add grid?
        self.plot_obj = self.ax.hexbin(self.X, self.Y, C=self.Z,
                                       cmap=self.cmap, vmin=self.p_min, vmax=self.p_max)

    def plot_margins(self):
        """
        Joint plot of heatmap (pcolormesh).
        Must input raw probabilities from H5_Pdist(p_units = 'raw').
        """
        # clean up infs and NaNs in Z
        Z = np.ma.masked_invalid(self.Z)
        # calc margin datasets
        x_proj = Z.sum(axis=0)
        y_proj = Z.sum(axis=1)
        # plot margins
        self.fig["x"].plot(self.X, self._normalize(x_proj), color=self.color)
        self.fig["y"].plot(self._normalize(y_proj), self.Y, color=self.color)

        # TODO: add functionality for scatter3d, use XY data to create gaussian_kde margins

    def _unpack_plot_options(self):
        """
        Unpack the plot_options kwarg dictionary.
        """
        # unpack plot options dictionary
        # TODO: put all in ax.set()?
        #for key, item in self.plot_options.items():
        for key, item in self.kwargs.items():
            if key == "xlabel":
                self.ax.set_xlabel(item)
            if key == "ylabel":
                self.ax.set_ylabel(item)
            if key == "xlim":
                self.ax.set_xlim(item)
                if self.jointplot:
                    self.fig["x"].set_xlim(item)
            if key == "ylim":
                self.ax.set_ylim(item)
                if self.jointplot:
                    self.fig["y"].set_ylim(item)
            if key == "title":
                self.ax.set_title(item)
            if key == "suptitle":
                plt.suptitle(item)
            # TODO: add grid to cli plot formatting args?
            if key == "grid" and item is True:
                self.ax.grid(item, alpha=0.5)
                if self.jointplot:
                    # grid the margins
                    for ax in ["x", "y"]:
                        self.fig[ax].grid(item, alpha=0.5)
            if key == "minima": # TODO: this is essentially bstate, also put maxima?
                # reorient transposed hist matrix
                Z = np.rot90(np.flip(self.Z, axis=0), k=3)
                # get minima coordinates index (inverse maxima since min = 0)
                maxima = np.where(1 / Z ==  np.amax(1 / Z, axis=(0, 1)))
                # plot point at x and y bin midpoints that correspond to mimima
                self.ax.plot(self.X[maxima[0]], self.Y[maxima[1]], 'ko')
                print(f"Minima: ({self.X[maxima[0]][0]}, {self.Y[maxima[1]][0]})")

    # TODO: cbar issues with 1d plots
    def plot(self, cbar=True):
        """
        Main public method.
        Master plotting run function
        Parse plot type and add cbars/tightlayout/plot_options/smoothing

        Parameters
        ----------
        cbar : bool
            Whether or not to include a colorbar.
        """
        # special settings for joint plots
        if self.jointplot:
            # can't use custom ax objects with jointplot, must create new fig and ax using mosaic
            if self.ax:
                message = "Can't use custom mpl axes objects with jointplot option, " + \
                          "creating new fig and ax using mpl subplot_mosaic."
                warn(message)
            # right now can't handle scatter with joint plot, it wouldn't be kT but standard hist
            if self.plot_mode == "scatter3d":
                warn("EXITING: Currently can't use `--plot-mode scatter3d` with `--jointplot`.")
                sys.exit(0)
            # since jointplot starts with raw probabilities
            # need to figure out what p_units are needed
            # only if p_units is definied (e.g. H5_Pdist args are in place)
            try:
                self.p_units
                self.T
            # if H5_Pdist args not in place, use default
            # TODO: this warning always pops up even is args.p_units is set
            except AttributeError as e:
                warn(f"{e}: Defaulting to 'kT' probability units.")
                # self.p_units does not exist, default to kT
                self.p_units = "kT"
            # if H5_Pdists args were in place and changed to "raw" for jointplots
            if self.p_units == "raw":
                # return to requested p_units
                self.p_units = self.requested_p_units

            # set figure attribute to be a mosaic plot
            self.fig = plt.figure(layout="tight").subplot_mosaic(
                                                    """
                                                    x..
                                                    Hyc
                                                    """,
                                                    # set the height ratios between the rows
                                                    height_ratios=[1, 3.5],
                                                    # set the width ratios between the columns
                                                    width_ratios=[3.5, 1, 0.25],
                                                    )
            self.cax = self.fig["c"]
            self.ax = self.fig["H"]
            # plot margins first from raw probabilities
            self.plot_margins()
            # calc normalized hist using updated p_units (could also do Z[Z != np.isinf])
            self.Z = self._normalize(np.ma.masked_invalid(self.Z))

            # add formatting jointplot options
            # remove redundant tick labels
            # self.fig["x"].set_xticks([])
            # self.fig["y"].set_yticks([])
            self.fig["x"].set_xticklabels([])
            self.fig["y"].set_yticklabels([])
            # put pmax as lims on margin plots
            self.fig["x"].set_ylim(self.p_min, self.p_max)
            self.fig["y"].set_xlim(self.p_min, self.p_max)

        else:
            if self.ax is None:
                self.fig, self.ax = plt.subplots()
            else:
                self.fig = plt.gcf()

        # smooth the data if specified
        if self.smoothing_level:
            self.Z = scipy.ndimage.gaussian_filter(self.Z, sigma=self.smoothing_level)
            # get rid of any negatives --> 0
            #self.Z[self.Z < 0] = np.inf
            #self.Z[self.Z == np.inf] = 0
            self.Z[self.Z < 0] = 0

        # get contour levels if needed
        if self.plot_mode in ["contour", "contour_l", "contour_f", "hist_l"]:
            self._get_contour_levels()

        if self.plot_mode == "contour":
            self.plot_contour_l()
            self.plot_contour_f()

        elif self.plot_mode == "contour_l":
            self.plot_contour_l()

        elif self.plot_mode == "contour_f":
            self.plot_contour_f()

        # TODO: auto label WE iterations on evolution? (done via __main__ right now)
        elif self.plot_mode == "hist" or self.plot_mode == "hist_l":
            if self.plot_mode == "hist_l":
                self.plot_contour_l()
            # I run into this error when I run something like instant with 
            # the h5 but didn't adjust the plot mode to something like line
            try:
                self.plot_hist()
            except (TypeError,ValueError) as e:
                # TODO: put the text into logger?
                print(f"{e}: Did you mean to use the default 'hist' plot mode?")
                print("Perhaps you need to define another dimension via '--Yname'?")
                sys.exit()

        elif self.plot_mode == "bar":
            self.plot_bar()
            Warning("'bar' plot_mode is still under development")

        elif self.plot_mode == "line":
            self.plot_line()
            self.ax.set_ylabel(self.cbar_label)

        elif self.plot_mode == "scatter3d":
            self.plot_scatter3d()

        elif self.plot_mode == "hexbin3d":
            self.plot_hexbin3d()

        # error if unknown plot_mode
        else:
            raise ValueError(f"plot_mode = '{self.plot_mode}' is not valid.")

        # TODO: can this work with non H5_Pdist input?
        # if self.Xname == "pcoord":
        #     self.ax.set_xlabel(f"Progress Coordinate {self.Xindex}")
        # if self.Yname == "pcoord":
        #     self.ax.set_ylabel(f"Progress Coordinate {self.Yindex}")

        # don't add cbar if not specified or if using a 1D plot
        if cbar and self.plot_mode not in ["line", "bar", "contour_l"]:
            self.add_cbar(cax=self.cax)

        # TODO: update to just unpack kwargs
        if self.kwargs is not None:
            self._unpack_plot_options()

        # fig vs plt shouldn't matter here (needed to go plt for mosaic)
        #self.fig.tight_layout()
        plt.tight_layout()
