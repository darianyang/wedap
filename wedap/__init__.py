"""
Author: Darian T. Yang
Date of Creation: September 13th, 2021 

Description:

"""

# Welcome to the wedap module!
from .h5_pdist import *
from .h5_plot import *
from .h5_gif import *

# Friendly, array-first aliases. H5_Pdist/H5_Plot are named for their H5 origin,
# but H5_Plot also accepts plain numpy X/Y/Z arrays directly, e.g.
#     X, Y, Z = wedap.Pdist("west.h5", "average").pdist()
#     wedap.Plot(X, Y, Z, plot_mode="hist").plot()
# Note: numpy arrays should be passed to Plot/H5_Plot (X, Y, Z), NOT to
# Pdist/H5_Pdist, whose first positional argument is the h5 file path.
Pdist = H5_Pdist
Plot = H5_Plot
