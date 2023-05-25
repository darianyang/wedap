"""
TODO: options for custom weights just like Sinan's PR for w_pdist?
TODO: add styles for 1 col, 2 col, and poster figures.
"""

# import wedap
# import matplotlib.pyplot as plt

# plt.style.use("styles/default.mplstyle")

# # TODO: auto ignore auxy when using 1d
# data_options = {"h5" : "data/west_c2x_4b.h5",
#                 "Xname" : "1_75_39_c2",
#                 "data_type" : "average",
#                 #"weighted" : True,
#                 #"p_min" : 1,
#                 #"p_max" : 5,
#                 "p_units" : "raw",
#                 #"first_iter" : 400,
#                 "last_iter" : 100, 
#                 #"last_iter" : 20, 
#                 #"bins" : 100, # note bins affects contour quality
#                 "plot_mode" : "line",
#                 "weighted" : False,
#                 #"plot_mode" : "hexbin3d",
#                 }

# # TODO: default to aux for labels if available or pcoord dim if None
# plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
#                 #"ylabel" : "RMSD ($\AA$)", 
#                 #"ylabel" : "WE Iteration", 
#                 "ylabel" : "weighted P on log scale",
#                 #"ylabel" : "$-RT\ \ln\, P\ (kcal\ mol^{-1})$", 
#                 "xlabel" : "Helical Angle (°)",
#                 #"ylabel" : "3 Point Angle (°)",
#                 #"xlabel" : "E175 $\chi_1$",
#                 #"ylabel" : "E175 $\chi_2$",
#                 #"title" : "2KOD 20-100° WE i400-500",
#                 #"title" : "1A43 20-100° WE i490-500",
#                 #"title" : "2KOD 10µs RMSD WE",
#                 #"ylim" : (1,7),
#                 #"ylim" : (470,500),
#                 #"xlim" : (10, 120),
#                 #"xlim" : (17, 22),
#                 #"ylim" : (30, 100),
#                 #"xlim" : (-180,180),
#                 #"ylim" : (-180,180),
#                 #"xlim" : (20,100),
#                 #"grid" : True,
#                 #"minima" : True, 
#                 # TODO: diagonal line option
#                 }


# # time the run
# import timeit
# start = timeit.default_timer()

# # TODO: eventually use this format to write unit tests of each pdist method

# #X, Y, Z = H5_Plot(**data_options)
# #H5_Plot(X, Y, Z).plot_hist_2d()

# # TODO: I should be able to use the classes sepertely or together
# #H5_Plot(plot_options=plot_options, **data_options).plot_contour()


# plot1 = wedap.H5_Plot(**data_options)
# plot1.plot()
# #plot1.ax.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
# #plot1.ax.set_yscale("log")
# # plot2 = wedap.H5_Plot(plot_options=plot_options, ax=plot1.ax, **data_options)
# # plot2.plot()

# # fig, ax = plt.subplots()
# # plot1 = wedap.H5_Plot(plot_options=plot_options, ax=ax, **data_options)
# # plot1.plot()
# # plot2 = wedap.H5_Plot(plot_options=plot_options, ax=ax, **data_options)
# # plot2.plot()

# # wedap = H5_Pdist(**data_options)
# # X, Y, Z = wedap.pdist()
# #plt.pcolormesh(X, Y, Z)

# # TODO: put this in a seperate file?
# # data for tests
# # mode = "instant"
# # for pcoord in ["pcoord", "dihedral_2"]:
# #     for pcoord2 in ["dihedral_3", "dihedral_4"]:
# #         pdist = H5_Pdist("data/p53.h5", mode, Xname=pcoord, Yname=pcoord2)
# #         X, Y, Z = pdist.pdist()
# #         np.savetxt(f"tests/{mode}_{pcoord}_{pcoord2}_X.txt", X)
# #         np.savetxt(f"tests/{mode}_{pcoord}_{pcoord2}_Y.txt", Y)
# #         np.savetxt(f"tests/{mode}_{pcoord}_{pcoord2}_Z.txt", Z)

# #wedap.plot()
# #plt.savefig("west_c2.png")
# #print(wedap.auxnames)

# # TODO: test using different p_max values and bar

# # 1D example
# # pdist = H5_Pdist("data/west_i200.h5", X="1_75_39_c2", **data_options)
# # X, Y = pdist.run()
# # plt.plot(X, Y)

# stop = timeit.default_timer()
# execution_time = stop - start
# print(f"Executed in {execution_time:04f} seconds")

# plt.show()

# TODO: build an tracer function that finds the iter,seg,frame(s) for an input hist bin

import wedap

x, y, _ = wedap.H5_Pdist(h5="data/west_c2x_4b.h5", data_type="average", p_units="raw", weighted=True).pdist()

# plot the 1d
import matplotlib.pyplot as plt
plt.plot(x, y)
plt.yscale("log")
plt.show()