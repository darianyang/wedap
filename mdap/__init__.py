"""
Author: Darian T. Yang
Date of Creation: May 2nd, 2023 

Description:

"""

# Welcome to the mdap module!
from .md_pdist import *
from .md_plot import *

# Friendly aliases mirroring wedap.Pdist / wedap.Plot, e.g.
#     X, Y, Z = mdap.Pdist(Xname="data.dat").pdist()
#     mdap.Plot(X=X, Y=Y, Z=Z, plot_mode="hist").plot()
Pdist = MD_Pdist
Plot = MD_Plot
