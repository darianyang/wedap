#import wedap
#import matplotlib.pyplot as plt
#import numpy as np

import matplotlib
matplotlib.use('agg')

from wedap.tests.test_h5_plot import plot_data_gen

h5 = "wedap/data/p53.h5"

# # 1 dataset plots
# for dt in ["evolution"]:
#     for pm in ["hist"]:
#         for x in ["pcoord", "dihedral_2"]:
#             plot_data_gen(h5, dt, pm, x, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}.npy")
            #plot_data_gen(h5, dt, pm, x, show=True)
# for dt in ["average", "instant"]:
#     for pm in ["line"]:
#         for x in ["pcoord", "dihedral_2"]:
#             plot_data_gen(h5, dt, pm, x, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}.npy")
#             #plot_data_gen(h5, dt, pm, x, show=True)

# # 2 dataset plots
# for dt in ["average", "instant"]:
#     for pm in ["hist", "contour"]:
#         for x, y in [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]]:
#             plot_data_gen(h5, dt, pm, x, y, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}.npy")
#             #plot_data_gen(h5, dt, pm, x, y, show=True)

# 2 dataset plots with joint plots (TODO)
# for jp in [True, False]:
#     for dt in ["average", "instant"]:
#         for pm in ["hist", "contour"]:
#             for x, y in [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]]:
#                 plot_data_gen(h5, dt, pm, x, y, jointplot=jp, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}_jp{jp}.npy")

# # 3 dataset plots
# for dt in ["average", "instant"]:
#     for pm in ["scatter3d", "hexbin3d"]:
#         for x, y, z in [["pcoord", "dihedral_2", "dihedral_3"], 
#                         ["dihedral_2", "pcoord", "dihedral_3"], 
#                         ["dihedral_2", "dihedral_3", "pcoord"]]:
#             plot_data_gen(h5, dt, pm, x, y, z, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}_{z}.npy")
#             #plot_data_gen(h5, dt, pm, x, y, z, show=True)