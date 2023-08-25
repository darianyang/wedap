"""
Plot MD pdists.

TODO: specify option for:
    timeseries (option for KDE side plot and option for stdev vs all reps)
    pdist (1D hist, 1D KDE, + others from H5_Plot)
    others?
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from warnings import warn
from numpy import inf

from .md_pdist import MD_Pdist

# relative import of wedap H5_Plot class (TODO: alt import?)
from wedap import H5_Plot

# subclass H5_Plot to have access to those pdist plots
# also multiple inheritance for features from MD_Pdist as well
class MD_Plot(H5_Plot, MD_Pdist):
    """
    These methods provide various plotting options for pdist data.
    """
    def __init__(self, *args, **kwargs):

        # # if Xname or Yname is input, first generate pdist
        # if "Xname" in kwargs or "Yname" in kwargs or "Zname" in kwargs:
        #     # initialize md pdist
        #     MD_Pdist.__init__(self, *args, **kwargs)
        #     # generate xyz arrays
        #     X, Y, Z = self.pdist()
        #     H5_Plot.__init__(self, X, Y, Z, *args, **kwargs)

        # # otherwise run with regular h5 plot settings
        # else:
        #     # #H5_Plot.__init__(self, *args, **kwargs)
        #     # #super().__init__(*args, **kwargs)
        #     # # initialize md pdist
        #     # MD_Pdist.__init__(self, *args, **kwargs)
        #     # # generate xyz arrays
        #     # X, Y, Z = self.pdist()
        #     # H5_Plot.__init__(self, X, Y, Z, *args, **kwargs)

        # for jointplots, save original p_units and run pdist with raw
        og_p_units = kwargs["p_units"]
        if kwargs["jointplot"]:
            kwargs["p_units"] = "raw"
        # initialize md pdist
        MD_Pdist.__init__(self, *args, **kwargs)
        # generate xyz arrays
        X, Y, Z = self.pdist()

        if kwargs["jointplot"]:
            kwargs["p_units"] = og_p_units
        print(f"NOP: {og_p_units}")
        print(kwargs["p_units"])
        H5_Plot.__init__(self, X, Y, Z, *args, **kwargs)


    # def __init__(self, X=None, Y=None, Z=None, plot_mode="hist", cmap="viridis", smoothing_level=None,
    #     color="tab:blue", ax=None, plot_options=None, p_min=None, p_max=None, contour_interval=1,
    #     cbar_label=None, cax=None, jointplot=False, *args, **kwargs):
    #     """
    #     Plotting of pdists generated from MD calculated datasets.

    #     Parameters
    #     ----------
    #     X, Y : arrays
    #         x and y axis values, and if using aux_y or evolution (with only aux_x), also must input Z.
    #     Z : 2darray
    #         Z is a 2-D matrix of the normalized histogram values.
    #     plot_mode : str
    #         TODO: update and expand. Can be 'hist' (default), 'contour', 'line', 'scatter3d'.
    #     cmap : str
    #         Can be string or cmap to be input into mpl. Default = viridis.
    #     smoothing_level : float
    #         Optionally add gaussian noise to smooth Z data. A good value is around 0.4 to 1.0.
    #     color : str
    #         Color for 1D plots.
    #     ax : mpl axes object
    #     plot_options : kwargs dictionary
    #         Include mpl based plot options (e.g. xlabel, ylabel, ylim, xlim, title).
    #     p_min : int
    #         The minimum probability limit value.
    #     p_max : int
    #         The maximum probability limit value.
    #     contour_interval : int
    #         Interval to put contour levels if using 'contour' plot_mode.
    #     cbar_label : str
    #         Label for the colorbar.
    #     cax : MPL axes object
    #         Optionally define axes object to place colorbar.
    #     jointplot : bool
    #         Whether or not to include marginal plots. Note to use this argument, 
    #         probabilities for Z or from H5_Pdist must be in `raw` p_units.
    #     ** args
    #     ** kwargs
    #     """
    #     # include the init args for H5_Pdist
    #     # TODO: how to make some of the args optional if I want to use classes seperately?
    #     #super().__init__(*args, **kwargs)

    #     self.ax = ax
    #     self.smoothing_level = smoothing_level
    #     self.jointplot = jointplot

    #     # TODO: option if you want to generate pdist
    #     # also need option of just using the input X Y Z args
    #     # or getting them from w_pdist h5 file, or from H5_Pdist output file
    #     # user inputs XYZ
    #     # if X is None and Y is None and Z is None:
    #     #     super().__init__(*args, **kwargs)
    #     #     # save the user requested p_units and changes p_units to raw
    #     #     if self.jointplot:
    #     #         self.requested_p_units = self.p_units
    #     #         kwargs["p_units"] = "raw"
    #     #     # will be re-normed later on
    #     #     X, Y, Z = H5_Pdist(*args, **kwargs).pdist()

    #     self.X = X
    #     self.Y = Y
    #     self.Z = Z

    #     self.p_min = p_min
    #     self.p_max = p_max
    #     self.contour_interval = contour_interval

    #     self.plot_mode = plot_mode
    #     self.cmap = cmap
    #     self.color = color # 1D color
    #     self.plot_options = plot_options

    #     # TODO: not compatible if inputing data instead of running pdist
    #     # try checking for the variable first, could use a t/e block
    #     #if self.p_units in locals():
    #     # if self.p_units == "kT":
    #     #     self.cbar_label = "$-\ln\,P(x)$"
    #     # elif self.p_units == "kcal":
    #     #     self.cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"

    #     # user override None cbar_label TODO
    #     if cbar_label:
    #         self.cbar_label = cbar_label
    #     else:
    #         self.cbar_label = "-ln P(x)"
    #     self.cax = cax

    #     super().__init__(*args, **kwargs)