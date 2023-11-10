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
#             plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, Xname=x, show=True)
#                           out=f"wedap/tests/data/plot_{dt}_{pm}_{x}.npy")
# for dt in ["average", "instant"]:
#     for pm in ["line"]:
#         for x in ["pcoord", "dihedral_2"]:
#             plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, Xname=x, #show=True)
#                           out=f"wedap/tests/data/plot_{dt}_{pm}_{x}.npy")

# 2 dataset plots
# for dt in ["average", "instant"]:
#     for pm in ["hist", "contour"]:
#         for x, y in [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]]:
#             plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, Xname=x, Yname=y, #show=True) 
#                           out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}.npy")

# # 3 dataset plots
# for dt in ["average", "instant"]:
#     for pm in ["scatter3d", "hexbin3d"]:
#         for x, y, z in [["pcoord", "dihedral_2", "dihedral_3"], 
#                         ["dihedral_2", "pcoord", "dihedral_3"], 
#                         ["dihedral_2", "dihedral_3", "pcoord"]]:
#             #plot_data_gen(h5, dt, pm, x, y, z, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}_{z}.npy")
#             plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, Xname=x, Yname=y, Zname=z, show=True)

# tests for evolution arg options
# for dt in ["evolution"]:
#     for pm in ["hist"]:
#         # tests for first_iter, last_iter, step_iter
#         for fi, li, si in [[1, 15, 1], 
#                            [3, None, 1], 
#                            [5, 15, 3]]:
#             plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, first_iter=fi, last_iter=li, step_iter=si, 
#                          out=f"wedap/tests/data/plot_{dt}_{pm}_fi{fi}_li{li}_si{si}.npy")
#             #plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, first_iter=fi, last_iter=li, step_iter=si, show=True)

#         # test bins and hrx
#         bins = 50
#         hrx = [0, 8]
#         plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, bins=bins, histrange_x=hrx,
#                      out=f"wedap/tests/data/plot_{dt}_{pm}_bins{bins}_hrx{hrx[0]}-{hrx[1]}.npy")
#         #plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, bins=bins, histrange_x=hrx, show=True)

# tests for average arg options
# for dt in ["average"]:
#     for pm in ["hist"]:
#         # tests for first_iter, last_iter, step_iter
#         for fi, li, si in [[1, 15, 1], 
#                            [3, None, 1], 
#                            [5, 15, 3]]:
#             plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, first_iter=fi, last_iter=li, step_iter=si, 
#                           Yname="pcoord", Yindex=1, #show=True)
#                           out=f"wedap/tests/data/plot_{dt}_{pm}_fi{fi}_li{li}_si{si}.npy")

#         # test bins and hrx/hry
#         bins = 50
#         hrx = [0, 8]
#         hry = [5, 35]
#         plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, bins=bins, histrange_x=hrx, histrange_y=hry,
#                       Yname="pcoord", Yindex=1, #show=True)
#                       out=f"wedap/tests/data/plot_{dt}_{pm}_bins{bins}_hrx{hrx[0]}-{hrx[1]}_hry{hry[0]}-{hry[1]}.npy")


##########################################
################## TODO ##################
##########################################
# # TODO: tests for p_units (all versions - maybe better with pdist test) and with --weighted
# # TODO: tests for 3d and 4d proj plots
# 2 dataset plots with joint plots (TODO)
# for jp in [True, False]:
#     for dt in ["average", "instant"]:
#         for pm in ["hist", "contour"]:
#             for x, y in [["pcoord", "dihedral_2"], ["dihedral_2", "pcoord"]]:
#                 plot_data_gen(h5=h5, data_type=dt, plot_mode=pm, Xname=x, Yname=y,, jointplot=jp, out=f"wedap/tests/data/plot_{dt}_{pm}_{x}_{y}_jp{jp}.npy")