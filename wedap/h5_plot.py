"""
Main plotting class of wedap.
Plot all of the datasets generated with h5_pdist.

# TODO: include trace and plot walker functionality with search_aux

    # TODO: all plotting options with test.h5, compare output
        # 1D Evo, 1D and 2D instant and average
        # optional: diff max_iter and bins args

TODO: make mpl style options path set up at install

TODO: maybe make methods for the following plots:
        '``contourf``--plot contour levels. '
        '``histogram``--plot histogram. '
        '``lines``--plot contour lines only. '
        '``contourf_l``--plot contour levels and lines. '
        '``histogram_l``--plot histogram and contour lines. ',
        option - with and without side histograms
        mpl mosaic options

TODO: plot clustering centroids option
      can then grab the search_aux at the centroid

TODO: bin visualizer
TODO: and maybe show the trajectories as just dots
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from warnings import warn
from numpy import inf

from .h5_pdist import *

# TODO: add search aux as a class method with seperate args?
    # have trace options in args for trace iter,wlk and x,y vals 
# TODO: method for each type of plot
# TODO: could subclass the H5_Pdist class, then use this as the main in wedap.py
class H5_Plot(H5_Pdist):

    def __init__(self, X=None, Y=None, Z=None, plot_mode="hist2d", cmap="viridis", 
        color="tab:blue", ax=None, plot_options=None, p_min=0, p_max=None, cbar_label=None,
        data_smoothing_level=None, curve_smoothing_level=None, *args, **kwargs):
        """
        Plotting of pdists generated from H5 datasets.TODO: update docstrings

        Parameters
        ----------
        # TODO: add/fix data smoothing
        data_smoothing_level : float
            A good value is around 0.4.
        curve_smoothing_level : float
            A good value is around 0.4.
        x, y : ndarray
            x and y axis values, and if using aux_y or evolution (with only aux_x), also must input Z.
        args_list : argparse.Namespace
            Contains command line arguments passed in by user.
        Z : ndarray
            Z is a 2-D matrix of the normalized histogram values.
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
                Can be 'kT' (default) or 'kcal'. kT = -lnP, kcal/mol = -RT(lnP), 
                where RT = 0.5922 at 298K.
            cmap : str
                Colormap option, default = viridis.
            **plot_options : kwargs
        """
        # include the init args for H5_Pdist
        # TODO: how to make some of the args optional if I want to use classes seperately?
        #super().__init__(*args, **kwargs)

        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig = plt.gcf()
            self.ax = ax

        self.data_smoothing_level = data_smoothing_level
        self.curve_smoothing_level = curve_smoothing_level

        # TODO: option if you want to generate pdist
        # also need option of just using the input X Y Z args
        # or getting them from w_pdist h5 file, or from H5_Pdist output file
        # user inputs XYZ
        if X is None and Y is None and Z is None:
            super().__init__(*args, **kwargs)
            X, Y, Z = H5_Pdist(*args, **kwargs).pdist()

        self.X = X
        self.Y = Y
        self.Z = Z

        self.p_min = p_min
        self.p_max = p_max

        self.plot_mode = plot_mode
        self.cmap = cmap
        self.color = color # 1D color
        self.plot_options = plot_options

        # TODO: not compatible if inputing data instead of running pdist
        # if self.p_units == "kT":
        #     self.cbar_label = "$-\ln\,P(x)$"
        # elif self.p_units == "kcal":
        #     self.cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"

        # user override None cbar_label TODO
        if cbar_label:
            self.cbar_label = cbar_label
        else:
            self.cbar_label = "-ln P(x)"

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

    def add_cbar(self):
        cbar = self.fig.colorbar(self.plot)
        # TODO: lines on colorbar?
        # TODO: related, make a discrete colorbar/mapping for hist2d?
        #if lines:
        #    cbar.add_lines(lines)
        # TODO: move labelpad here to style
        cbar.set_label(self.cbar_label, labelpad=14)

        # allow for cbar object manipulation (e.g. removal in movie)
        self.cbar = cbar
    
    def plot_hist2d(self):
        # 2D heatmaps
        # TODO: westpa makes these the max to keep the pdist shape
        # if self.p_max:
        #     self.Z[self.Z > self.p_max] = inf
        self.plot = self.ax.pcolormesh(self.X, self.Y, self.Z, cmap=self.cmap, 
                                       shading="auto", vmin=self.p_min, vmax=self.p_max)

    def plot_contour(self):
        # TODO: seperate functions for contourf and contourl?
            # then can use hist and contourl
        # 2D contour plots
        if self.p_max is None:
            warn("With 'contour' plot_type, p_max should be set. Otherwise max Z is used.")
            levels = np.arange(self.p_min, np.max(self.Z[self.Z != np.inf ]), 1)
        elif self.p_max <= 1:
            levels = np.arange(self.p_min, self.p_max + 0.2, 0.2)
        else:
            levels = np.arange(self.p_min, self.p_max + 1, 1)

        # TODO: better implementation of this
        if self.curve_smoothing_level:
            # TODO: gets rid of nan, I don't think needed here since _smooth does this
            #self.Z_curves[np.isnan(self.Z_curves)] = np.max(self.Z)*2
            self.lines = self.ax.contour(self.X, self.Y, self.Z_curves, levels=levels, colors="black", linewidths=1)
        else:
            self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=levels, colors="black", linewidths=1)

        self.plot = self.ax.contourf(self.X, self.Y, self.Z, levels=levels, cmap=self.cmap)

    def plot_bar(self):
        # 1D data
        # recover the pdf from the -ln P
        # TODO: does this account for p_max naturally?
        self.ax.bar(self.X, np.exp(-self.Y), color=self.color)
        #self.ax.bar(self.X, self.Y, color=self.color)
        self.ax.set_ylabel("P(x)")

    def plot_hist1d(self):
        # 1D data : TODO: not working currently
        # recover the pdf from the -ln P
        # TODO: does this account for p_max naturally?
        # TODO: I can get the raw data and then get the counts right from XYZ functions
        self.ax.hist(self.X, self.Y)
        self.ax.set_ylabel("P(x)")

    def plot_line(self):
        # 1D data
        #if self.p_max:
        #    self.Y[self.Y > self.p_max] = inf
        self.ax.plot(self.X, self.Y, color=self.color)
        self.ax.set_ylabel(self.cbar_label)
    
    def plot_scatter3d(self):
        self.plot = self.ax.scatter(self.X, self.Y, c=self.Z, cmap=self.cmap)

    def _unpack_plot_options(self):
        """
        Unpack the plot_options kwarg dictionary.
        """
        # unpack plot options dictionary
        # TODO: put all in ax.set()?
        for key, item in self.plot_options.items():
            if key == "xlabel":
                self.ax.set_xlabel(item)
            if key == "ylabel":
                self.ax.set_ylabel(item)
            if key == "xlim":
                self.ax.set_xlim(item)
            if key == "ylim":
                self.ax.set_ylim(item)
            if key == "title":
                self.ax.set_title(item)
            if key == "grid" and key is True:
                self.ax.grid(item, alpha=0.5)
            if key == "minima": # TODO: this is essentially bstate, also put maxima?
                # reorient transposed hist matrix
                Z = np.rot90(np.flip(self.Z, axis=0), k=3)
                # get minima coordinates index (inverse maxima since min = 0)
                maxima = np.where(1 / Z ==  np.amax(1 / Z, axis=(0, 1)))
                # plot point at x and y bin midpoints that correspond to mimima
                self.ax.plot(self.X[maxima[0]], self.Y[maxima[1]], 'ko')
                print(f"Minima: ({self.X[maxima[0]][0]}, {self.Y[maxima[1]][0]})")

    # TODO: i think the data smoothing works but not the curve
    def _smooth(self):
        if self.data_smoothing_level is not None:
            self.Z[np.isnan(self.Z)] = np.nanmax(self.Z)
            self.Z = scipy.ndimage.filters.gaussian_filter(self.Z, 
                                self.data_smoothing_level)
        if self.curve_smoothing_level is not None:
            self.Z_curves[np.isnan(self.Z_curves)] = np.nanmax(self.Z_curves)
            self.Z_curves = scipy.ndimage.filters.gaussian_filter(self.Z_curves, 
                                self.curve_smoothing_level)
        self.Z[np.isnan(self.Z)] = np.nan 
        #self.Z_curves[np.isnan(self.Z)] = np.nan 
        #self.Z_curves[np.isinf(self.Z_curves)] = np.max(self.Z)*2
        # TODO: are any of these needed, the Z_curve seems to not change much

    def plot(self):
        """
        Main public method.
        master plotting run function
        here can parse plot type and add cbars/tightlayout/plot_options/smoothing

        TODO: some kind 1d vs 2d indicator, then if not 1d plot cbar
        """
        if self.plot_mode == "contour":
            # Do data smoothing. We have to make copies of the array so that
            # the data and curves can have different smoothing levels.
            self.Z_curves = np.copy(self.Z)
            self._smooth()
            self.plot_contour()
            self.add_cbar()

        # TODO: auto label WE iterations on evolution?
        elif self.plot_mode == "hist2d":
            self.plot_hist2d()
            self.add_cbar()

        elif self.plot_mode == "bar":
            self.plot_bar()
            #self.ax.set_ylabel(self.cbar_label)

        elif self.plot_mode == "line":
            self.plot_line()
            self.ax.set_ylabel(self.cbar_label)

        elif self.plot_mode == "hist1d":
            self.plot_hist1d()
            self.ax.set_ylabel("Counts")

        elif self.plot_mode == "scatter3d":
            self.plot_scatter3d()
            self.add_cbar()

        # error if unknown plot_mode
        else:
            raise ValueError(f"plot_mode = '{self.plot_mode}' is not valid.")

        # TODO: can this work with non H5_Pdist input?
        # if self.Xname == "pcoord":
        #     self.ax.set_xlabel(f"Progress Coordinate {self.Xindex}")
        # if self.Yname == "pcoord":
        #     self.ax.set_ylabel(f"Progress Coordinate {self.Yindex}")

        if self.plot_options is not None:
            self._unpack_plot_options()         # TODO
        self.fig.tight_layout()
