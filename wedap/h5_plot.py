"""
Main plotting class of wedap.
Plot all of the datasets generated with h5_pdist.

# TODO: include trace and plot walker functionality with search_aux

    # TODO: all plotting options with test.h5, compare output
        # 1D Evo, 1D and 2D instant and average
        # optional: diff max_iter and bins args

TODO: add mpl style options

TODO: if there is x and y limits, use them to make the histogram bounds

TODO: maybe make methods for the following plots:
        '``contourf``--plot contour levels. '
        '``histogram``--plot histogram. '
        '``lines``--plot contour lines only. '
        '``contourf_l``--plot contour levels and lines. '
        '``histogram_l``--plot histogram and contour lines. ',
        option - with and without side histograms
        mpl mosaic options
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from warnings import warn
from numpy import inf

from h5_pdist import H5_Pdist

# TODO: add search aux as a class method with seperate args?
    # have trace options in args for trace iter,wlk and x,y vals 
# TODO: method for each type of plot
# TODO: could subclass the H5_Pdist class, then use this as the main in wedap.py
class H5_Plot(H5_Pdist):

    def __init__(self, X=None, Y=None, Z=None, plot_type="heat", cmap="viridis", ax=None, 
        plot_options=None, data_smoothing_level=None, curve_smoothing_level=None, 
        plot_mode="hist_2d", *args, **kwargs):
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
                Can be 'kT' (default) or 'kcal'. kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at 298K.
            cmap : str
                Colormap option, default = viridis.
            **plot_options : kwargs
        """
        # include the init args for H5_Pdist
        super().__init__(*args, **kwargs)

        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5,4))
        else:
            self.fig = plt.gcf()
            self.ax = ax

        self.data_smoothing_level = data_smoothing_level
        self.curve_smoothing_level = curve_smoothing_level

        # TODO: option if you want to generate pdist
        # also need option of just using the input X Y Z args
        # or getting them from w_pdist h5 file, or from H5_Pdist output file
        # TODO: 1D plot not working
        if plot_mode == "line_1d":
            X, Y = H5_Pdist(*args, **kwargs).pdist()
        elif X is None and Y is None and Z is None:
            X, Y, Z = H5_Pdist(*args, **kwargs).pdist()

        self.X = X
        self.Y = Y
        self.Z = Z

        self.plot_mode = plot_mode
        self.cmap = cmap
        self.plot_options = plot_options

        if self.p_units == "kT":
            self.cbar_label = "$\Delta F(\vec{x})\,/\,kT$" + "\n" + r"$\left[-\ln\,P(x)\right]$"
        elif self.p_units == "kcal":
            self.cbar_label = r"$\it{-RT}$ ln $\it{P}$ (kcal mol$^{-1}$)"

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

    def cbar(self):
        cbar = self.fig.colorbar(self.plot)
        # TODO: lines on colorbar?
        #if lines:
        #    cbar.add_lines(lines)
        cbar.set_label(self.cbar_label)
    
    def plot_hist_2d(self):
        # 2D heatmaps
        # if self.p_max:
        #     self.Z[self.Z > self.p_max] = inf
        self.plot = self.ax.pcolormesh(self.X, self.Y, self.Z, cmap=self.cmap, shading="auto", vmin=self.p_min, vmax=self.p_max)

    def plot_contour_2d(self):
        # 2D contour plots
        if self.p_max is None:
            warn("With 'contour' plot_type, p_max should be set. Otherwise max Z is used.")
            levels = np.arange(self.p_min, np.max(self.Z[self.Z != np.inf ]), 1)
        elif self.p_max <= 1:
            levels = np.arange(self.p_min, self.p_max + 0.1, 0.1)
        else:
            levels = np.arange(self.p_min, self.p_max + 1, 1)
        self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=levels, colors="black", linewidths=1)
        self.plot = self.ax.contourf(self.X, self.Y, self.Z, levels=levels, cmap=self.cmap)

    def plot_line_1d(self):
        # 1D data
        #if self.p_max:
        #    self.Y[self.Y > self.p_max] = inf
        self.ax.plot(self.X, self.Y)
        self.ax.set_ylabel("")

    def unpack_plot_options(self):
        """
        Unpack the plot_options kwarg dictionary.
        """
        # unpack plot options dictionary # TODO: update this for argparse?
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
            if key == "grid":
                self.ax.grid(item, alpha=0.5)
            if key == "minima": # TODO: this is essentially bstate, also put maxima?
                # reorient transposed hist matrix
                Z = np.rot90(np.flip(self.Z, axis=0), k=3)
                # get minima coordinates index (inverse maxima since min = 0)
                maxima = np.where(1 / Z ==  np.amax(1 / Z, axis=(0, 1)))
                # plot point at x and y bin midpoints that correspond to mimima
                self.ax.plot(self.X[maxima[0]], self.Y[maxima[1]], 'ko')
                print(f"Minima: ({self.X[maxima[0]][0]}, {self.Y[maxima[1]][0]})")


    # TODO: put in plotting class?
    # See AJD script for variable definitions
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
        self.Z_curves[np.isnan(self.Z)] = np.nan 

    # TODO
    # def _run_postprocessing(self):
    #     '''
    #     Run the user-specified postprocessing function.
    #     '''
    #     import importlib
    #     # Parse the user-specifed string for the module and class/function name.
    #     module_name, attr_name = self.args.postprocess_func.split('.', 1) 
    #     # import the module ``module_name`` and make the function/class 
    #     # accessible as ``attr``.
    #     attr = getattr(importlib.import_module(module_name), attr_name) 
    #     # Call ``attr``.
    #     attr()

    def plot(self):
        """
        Main public method.
        """
        if self.plot_mode == "contour":
            # Do data smoothing. We have to make copies of the array so that
            # the data and curves can have different smoothing levels.
            self.Z_curves = np.copy(self.Z)
            self._smooth()
            self.plot_contour_2d()
            self.cbar()

        if self.plot_mode == "hist_2d":
            self.plot_hist_2d()
            self.cbar()

        if self.plot_mode == "line_1d":
            self.plot_line_1d()

        self.unpack_plot_options()        
        self.fig.tight_layout()

    # TODO: master plotting run function
    # here can parse plot type and add cbars/tightlayout/plot_options/smoothing
