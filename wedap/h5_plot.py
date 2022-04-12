"""
Main plotting class of wedap.
Plot all of the datasets generated with h5_pdist.

# TODO: include trace and plot walker functionality with search_aux

    # TODO: all plotting options with test.h5, compare output
        # 1D Evo, 1D and 2D instant and average
        # optional: diff max_iter and bins args

TODO: add mpl style options
"""

import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


# TODO: method for each type of plot
# TODO: could subclass the H5_Pdist class, then use this as the main in wedap.py
class H5_plot:

    def __init__(self, ax=None):
        """
        Plotting of pdists generated from H5 datasets.
        """
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5,4))
        else:
            self.fig = plt.gcf()
            self.ax = ax

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
            # if self.p_max:
            #     norm_hist[norm_hist > self.p_max] = inf
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
