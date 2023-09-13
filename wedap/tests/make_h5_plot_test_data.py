import wedap

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np

from wedap.tests.test_h5_plot import plot_data_gen

# def plot_data_gen(h5, data_type, plot_mode, Xname, Yname=None, Zname=None, out=None):
#     """
#     Make plot and convert to npy binary data file.
#     """
#     plot = wedap.H5_Plot(h5=h5, data_type=data_type, plot_mode=plot_mode, 
#                         Xname=Xname, Yname=Yname, Zname=Zname)
#     plot.plot()
#     fig = plot.fig

#     #plt.show()
#     #import sys; sys.exit(0)

#     # If we haven't already shown or saved the plot, then we need to
#     # draw the figure first...
#     # tight since canvas is large
#     fig.tight_layout(pad=0)
#     fig.canvas.draw()

#     # Now we can save it to a numpy array.
#     data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     # save as np binary file
#     if out:
#         np.save(out, data)

h5 = "wedap/data/p53.h5"

# # 1 dataset plots
# for dt in ["evolution", "average", "instant"]:
#     for pm in ["line"]:
#         for x in ["pcoord", "dihedral_2"]:
#             plot_data_gen(h5, dt, pm, x, out=f"wedap/tests/data/data/plot_{dt}_{pm}_{x}.npy")

# # 2 dataset plots
# for dt in ["average", "instant"]:
#     for pm in ["hist", "contour"]:
#         for x, y in [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]]:
#             plot_data_gen(h5, dt, pm, x, y, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}.npy")

# 2 dataset plots with joint plots (TODO)
# for jp in [True, False]:
#     for dt in ["average", "instant"]:
#         for pm in ["hist", "contour"]:
#             for x, y in [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]]:
#                 plot_data_gen(h5, dt, pm, x, y, jointplot=jp, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}_jp{jp}.npy")

# # 3 dataset plots
# for dt in ["average", "instant"]:
#     for pm in ["scatter3d"]:
#         for x, y, z in [["pcoord", "dihedral_2", "dihedral_3"], 
#                         ["dihedral_2", "pcoord", "dihedral_3"], 
#                         ["dihedral_2", "dihedral_3", "pcoord"]]:
#             plot_data_gen(h5, dt, pm, x, y, z, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}_{z}.npy")