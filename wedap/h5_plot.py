"""
Main plotting class of wedap.
Plot all of the datasets generated with H5_Pdist.

* line -- plot 1D lines.
* hist -- plot histogram (default).
* hist_l -- plot histogram and contour lines.
* contour -- plot contour levels and lines.
* contour_f -- plot contour levels
* contour_l -- plot contour lines only.
* scatter3d -- plot 3 datasets in a scatter plot.

maybe a ridgeline plot?
- This would be maybe for 1D avg of every 100 iterations
- https://matplotlib.org/matplotblog/posts/create-ridgeplots-in-matplotlib/
Option to overlay different datasets, could be done easily with python but maybe a cli option?                

TODO: plot clustering centroids option?
      can then grab the search_aux at the centroid

TODO: bin visualizer? and maybe show the trajectories as just dots?
Add postprocess function for quick fixes when using CLI? see feplotter
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from warnings import warn
from numpy import inf
import importlib

from .h5_pdist import *

# TODO: maybe put the pdist object into the plot class and have this object be flexible
# so it could just be a pdist.h5 file from westpa or make your own
# or read in a pdist.h5 and create pdist object using that dataset

class H5_Plot(H5_Pdist):
    """
    These methods provide various plotting options for pdist data.
    """
    cbar_pad = 0.05
    def __init__(self, X=None, Y=None, Z=None, plot_mode="hist", cmap=None, smoothing_level=None,
        color=None, ax=None, p_min=None, p_max=None, contour_interval=1, contour_levels=None,
        cbar_label=None, cax=None, jointplot=False, data_label=None, proj3d=False, proj4d=False, 
        C=None, scatter_interval=10, scatter_s=1, hexbin_grid=100, linewidth=None, linestyle="-",
        postprocess_func=None, *args, **kwargs):
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
        data_label : str
            Optionally label the data, e.g. for multiple 1D plots.
        proj3d : bool
            Optionally use a 3d projection plot, defaut False.
            Only works with contour and scatter plots.
        proj4d : bool
            Optionally use a 4d projection plot, defaut False.
            Only works with scatter plots.
        C : array
            For color mapping of 3d projection plots.
        scatter_interval : int
            Interval for displaying scatter plot data, default 1.
        scatter_s : float
            Int for displaying scatter plot data marker size, default 1.
        hexbin_grid : int
            Determines gridsize for hexbin plots.
        linewidth : float
            Linewidth for 1D plots, contour lines, and hexbin edges.
        linestyle : str
            Linestyle for 1D plots, contour lines, and hexbin edges.
        postprocess_func : func
            User function to import.
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
        if X is None and Y is None and Z is None and C is None:
            super().__init__(*args, **kwargs)
            # save the user requested p_units and changes p_units to raw
            if self.jointplot:
                # will be re-normed later on
                X, Y, Z = self.pdist(normalize=False)
            # when requesting a projection plot with 4d cbar additional dataset
            elif proj4d:
                X, Y, Z, C = self.pdist()
            else:
                # TODO: tuple unpacking to deal with variable item return
                X, Y, Z = self.pdist()
        # need to set this when using mdap, shouldn't affect anything else
        # since joint plot dists must be changed from raw to requested p_units
        if self.jointplot and "p_units" in kwargs:
            self.requested_p_units = kwargs["p_units"]
        else:
            self.requested_p_units = "kT"

        self.X = X
        self.Y = Y
        self.Z = Z
        self.C = C

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
                #self.cbar_label = "-ln (P(x))"
            elif self.p_units == "kcal":
                self.cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"
            elif self.p_units == "raw":
                self.cbar_label = "Counts"
            elif self.p_units == "raw_norm":
                self.cbar_label = "Normalized Counts"
        # if using 3 datasets, put blank name as default cbar
        if self.plot_mode == "scatter3d" or self.plot_mode == "hexbin3d":
            self.cbar_label = ""
        # overwrite and apply cbar_label attr if available/specified
        if cbar_label:
            self.cbar_label = cbar_label

        self.cax = cax
        self.data_label = data_label
        self.proj3d = proj3d
        self.proj4d = proj4d
        self.scatter_interval = scatter_interval
        self.scatter_s = scatter_s
        self.hexbin_grid = hexbin_grid
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.postprocess_func = postprocess_func
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

    def add_cbar(self, cax=None, pad=0.05):
        """
        Add cbar.

        Parameters
        ----------
        cax : mpl cbar axis
            Optionally specify the cbar axis.
        pad : float
            cbar padding level.
        """
        # fig vs plt should be the same, tests run fine (needed to go plt for mosaic)
        #cbar = self.fig.colorbar(self.plot, cax=cax)
        cbar = plt.colorbar(self.plot_obj, cax=cax, pad=pad)
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
        Warning("contour_l lines are set to mpl defaults, set can be changed with `--color` or `--cmap`")
        # can control linewidths using rc params (lines.linewidths (default 1.5))
        if self.color:
            self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=self.contour_levels, 
                                         colors=self.color, linewidths=self.linewidth, linestyles=self.linestyle)
            #self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=[5], colors=self.color)
        else:
            self.lines = self.ax.contour(self.X, self.Y, self.Z, levels=self.contour_levels, 
                                         cmap=self.cmap, linewidths=self.linewidth, linestyles=self.linestyle)
        # set to call both lines and cbar plot_obj
        self.plot_obj = self.lines

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
        self.ax.bar(self.X, self.Y, color=self.color, label=self.data_label)
        self.ax.set_ylabel("P(x)")

    def plot_line(self):
        """
        1d line plot.
        """
        # 1D data
        if self.p_max:
            self.Y[self.Y > self.p_max] = inf
        self.ax.plot(self.X, self.Y, color=self.color, label=self.data_label, 
                     linewidth=self.linewidth, linestyle=self.linestyle)
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
        if self.proj3d or self.proj4d:
            if isinstance(self.C, np.ndarray):
                C = self.C[::interval]
            else:
                C = self.Z[::interval]
            self.plot_obj = self.ax.scatter(self.X[::interval], 
                                            self.Y[::interval], 
                                            self.Z[::interval],
                                            c=C,
                                            cmap=self.cmap, s=s,
                                            vmin=self.p_min, vmax=self.p_max)
        else:
            self.plot_obj = self.ax.scatter(self.X[::interval], 
                                            self.Y[::interval], 
                                            c=self.Z[::interval], 
                                            cmap=self.cmap, s=s,
                                            vmin=self.p_min, vmax=self.p_max)

    def plot_hexbin3d(self, gridsize=100):
        """
        Hexbin plot?
        """
        # TODO: test this and add grid?
        # reshape to 1D and then get rid of extra dimension
        self.X = np.squeeze(self.X.reshape(1, -1))
        self.Y = np.squeeze(self.Y.reshape(1, -1))
        self.Z = np.squeeze(self.Z.reshape(1, -1))
        #print(self.X.shape, self.Y.shape, self.Z.shape)
        self.plot_obj = self.ax.hexbin(self.X, self.Y, C=self.Z, gridsize=gridsize, 
                                       edgecolors=self.color, 
                                       linewidths=self.linewidth, linestyles=self.linestyle,
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
        self.fig["x"].plot(self.X, self._normalize(x_proj, self.p_units), color=self.color)
        self.fig["y"].plot(self._normalize(y_proj, self.p_units), self.Y, color=self.color)

        # TODO: add functionality for scatter3d, use XY data to create gaussian_kde margins

    def _unpack_plot_options(self):
        """
        Unpack the plot_options kwarg dictionary.
        """
        # unpack plot options dictionary making sure not None
        # TODO: put all in ax.set()?
        #for key, item in self.plot_options.items():
        for key, item in self.kwargs.items():
            if key == "xlabel" and item:
                self.ax.set_xlabel(item)
            if key == "ylabel" and item:
                self.ax.set_ylabel(item)
            if key == "xlim" and item:
                self.ax.set_xlim(item)
                if self.jointplot:
                    self.fig["x"].set_xlim(item)
            if key == "ylim"and item:
                self.ax.set_ylim(item)
                if self.jointplot:
                    self.fig["y"].set_ylim(item)
            if key == "title" and item:
                self.ax.set_title(item)
            if key == "suptitle" and item:
                plt.suptitle(item)
            if key == "grid" and item:
                self.ax.grid(item, alpha=0.5)
                if self.jointplot:
                    # grid the margins
                    for ax in ["x", "y"]:
                        self.fig[ax].grid(item, alpha=0.5)
            if key == "minima" and item: # TODO: this is essentially bstate, also put maxima?
                # reorient transposed hist matrix
                Z = np.rot90(np.flip(self.Z, axis=0), k=3)
                # get minima coordinates index (inverse maxima since min = 0)
                maxima = np.where(1 / Z ==  np.amax(1 / Z, axis=(0, 1)))
                # plot point at x and y bin midpoints that correspond to mimima
                self.ax.plot(self.X[maxima[0]], self.Y[maxima[1]], 'ko')
                print(f"Minima: ({self.X[maxima[0]][0]}, {self.Y[maxima[1]][0]})")
            
            # now allowing for a list of line inputs
            if key == "axvline" and item:
                # make into list if not already
                if not isinstance(item, list):
                    item = [item]
                # loop each list item and plot line
                for line in item:
                    self.ax.axvline(line, color=self.color, linewidth=self.linewidth, linestyle=self.linestyle)
            if key == "axhline" and item:
                # make into list if not already
                if not isinstance(item, list):
                    item = [item]
                # loop each list item and plot line
                for line in item:
                    self.ax.axhline(line, color=self.color, linewidth=self.linewidth, linestyle=self.linestyle)

    def _run_postprocessing(self):
        """
        Run the user-specified postprocessing function.
        """
        # Parse the user-specifed string for the module and class/function name.
        module_name, attr_name = self.postprocess_func.split('.', 1) 
        # import the module ``module_name`` and make the function/class 
        # accessible as ``attr``.
        #attr = getattr(importlib.import_module(module_name), attr_name) 
        attr = getattr(self.load_module(module_name, '.'), attr_name)
        # Call ``attr``.
        attr()

    @staticmethod
    def load_module(module_name, path=None):
        """Load and return the given module, recursively loading containing packages as necessary."""
        if module_name in sys.modules:
            return sys.modules[module_name]

        if path is None:
            return importlib.import_module(module_name)

        spec_components = list(reversed(module_name.split('.')))
        qname_components = []
        mod_chain = []
        while spec_components:
            next_component = spec_components.pop(-1)
            qname_components.append(next_component)

            try:
                parent = mod_chain[-1]
                path = parent.__path__
            except IndexError:
                parent = None

            qname = '.'.join(qname_components)

            if qname in sys.modules:
                module = sys.modules[qname]
            else:
                spec = importlib.machinery.PathFinder().find_spec(qname, path)

                if spec is None:
                    raise ImportError(f'No module named {qname}')

                module = importlib.util.module_from_spec(spec)

                if spec.name not in sys.modules:
                    sys.modules[spec.name] = module

                spec.loader.exec_module(module)

                # Make the module appear in the parent module's namespace
                if parent:
                    setattr(parent, next_component, module)

            mod_chain.append(module)

        return module

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
            # calc normalized hist using updated p_units (maybe could also do Z[Z != np.isinf])
            # but masked invalid takes care of Nan and inf converts to mask (--)
            #self.Z = self._normalize(self.Z[self.Z != np.inf], self.p_units)
            # TODO: now that normalize can take p_units, could be easier to not go
            #       back and forth with saving the requested p units and changing to raw
            self.Z = self._normalize(np.ma.masked_invalid(self.Z), self.p_units)

            # add formatting jointplot options
            # remove redundant tick labels
            # self.fig["x"].set_xticks([])
            # self.fig["y"].set_yticks([])
            self.fig["x"].set_xticklabels([])
            self.fig["y"].set_yticklabels([])
            # put pmax as lims on margin plots
            self.fig["x"].set_ylim(self.p_min, self.p_max)
            self.fig["y"].set_xlim(self.p_min, self.p_max)

        # 3dprojection test, not going to be compatible with jp
        elif self.proj3d or self.proj4d:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(projection='3d')
            # don't need 4d cbar when 3d projected
            if self.proj3d:
                cbar = False
                # but add cbar label to z axis
                self.ax.set_zlabel(self.cbar_label)
            # elif self.proj4d:
            #     self.ax.set_zlabel(self.cbar_label)

        else:
            if self.ax is None:
                self.fig, self.ax = plt.subplots()
            else:
                self.fig = plt.gcf()

        # smooth the data if specified
        if self.smoothing_level:
            # make into a smooth style e.g. contour with 0 as base instead of negative or inf
            #self.Z = np.ma.masked_invalid(self.Z)
            self.Z[self.Z < 0] = 0
            self.Z[self.Z == np.inf] = 0
            self.Z = scipy.ndimage.gaussian_filter(self.Z, sigma=self.smoothing_level)
            #self.Z[self.Z == 0] = np.inf
            # get rid of any negatives --> 0
            #self.Z[self.Z < 0] = np.inf
            #self.Z[self.Z == np.inf] = 0
            #self.Z[self.Z < 0] = 0

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
                print("Or if you want a 1D pdist, set --plot-mode or -pm to `line`.")
                sys.exit()

        elif self.plot_mode == "bar":
            self.plot_bar()
            Warning("'bar' plot_mode is still under development")

        elif self.plot_mode == "line":
            self.plot_line()
            #self.ax.set_ylabel(self.cbar_label)

        elif self.plot_mode == "scatter3d":
            self.plot_scatter3d(interval=self.scatter_interval, s=self.scatter_s)

        elif self.plot_mode == "hexbin3d":
            self.plot_hexbin3d(gridsize=self.hexbin_grid)

        # error if unknown plot_mode
        else:
            raise ValueError(f"plot_mode = '{self.plot_mode}' is not valid.")

        # TODO: can this work with non H5_Pdist input?
        # if self.Xname == "pcoord":
        #     self.ax.set_xlabel(f"Progress Coordinate {self.Xindex}")
        # if self.Yname == "pcoord":
        #     self.ax.set_ylabel(f"Progress Coordinate {self.Yindex}")

        # don't add cbar if not specified or if using a 1D plot
        if cbar and self.plot_mode not in ["line", "bar"]:
            self.add_cbar(cax=self.cax, pad=self.cbar_pad)

        # take kwargs and unpack to look for plot option items
        if self.kwargs is not None:
            self._unpack_plot_options()

        # optionally run post processing function
        # commented out, likely if you're using the API, no need for postproc
        # only gets called during CLI/GUI use
        # if self.postprocess_func is not None:
        #     self._run_postprocessing()

        # fig vs plt shouldn't matter here (needed to go plt for mosaic)
        #self.fig.tight_layout()
        plt.tight_layout()

if __name__ == "__main__":
    # testing of postprocess function
    import types
    x = types.SimpleNamespace()
    import os
    print(os.getcwd())
    x.postprocess_func = "postprocess_test.adjust_plot"
    #H5_Plot._run_postprocessing(x)

    #import importlib
    # # Parse the user-specifed string for the module and class/function name.
    # module_name, attr_name = x.postprocess_func.split('.', 1) 
    # # import the module ``module_name`` and make the function/class 
    # # accessible as ``attr``.
    # attr = getattr(importlib.import_module(module_name), attr_name) 
    # # Call ``attr``.
    # attr()
    