"""
Eventually incorporate into tests.

TODO: add styles for 1 col, 2 col, and poster figures.
"""
from h5_pdist import *
from h5_plot import *

# formatting
plt.style.use("default.mplstyle")

# TODO: auto ignore auxy when using 1d
data_options = {"h5" : "data/west_c2.h5",
                #"h5" : "data/multi_2kod.h5",
                #"h5" : "data/p53.h5",
                #"aux_x" : "1_75_39_c2",
                "aux_x" : "pcoord",
                #"aux_y" : "pcoord",
                #"aux_y" : "angle_3pt",
                #"aux_y" : "RoG",
                #"aux_y" : "XTAL_REF_RMS_Heavy",
                #"aux_x" : "rog",
                "aux_y" : "rms_bb_nmr",
                #"aux_y" : "rms_bb_xtal",
                #"aux_y" : "rms_m1_xtal",
                #"aux_x" : "M1_E175_chi2",
                #"aux_y" : "M2_E175_chi2",
                "data_type" : "instant",
                #"p_min" : 15,
                #"p_max" : 20,
                "p_units" : "kcal",
                #"first_iter" : 161,
                #"last_iter" : 161, 
                "bins" : 100, # note bins affects contour quality
                #"plot_mode" : "contour",
                #"cmap" : "gnuplot_r",
                "plot_mode" : "hist2d",
                #"plot_mode" : "line",
                #"data_smoothing_level" : 0.4,
                #"curve_smoothing_level" : 0.4,
                }

# TODO: default to aux for labels if available or pcoord dim if None
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                #"ylabel" : "RMSD ($\AA$)", 
                #"ylabel" : "WE Iteration", 
                #"ylabel" : "RMSD to XTAL ($\AA$)", 
                #"xlabel" : "Helical Angle (°)",
                "title" : "2KOD 20-100° WE",
                #"title" : "2KOD 10µs RMSD WE",
                #"xlim" : (1,7),
                #"ylim" : (1,7),
                #"xlim" : (0, 120),
                #"xlim" : (17, 22),
                #"ylim" : (30, 100),
                #"xlim" : (-180,180),
                #"ylim" : (-180,180),
                #"xlim" : (20,100),
                "grid" : True,
                #"minima" : True, 
                # TODO: diagonal line option
                }


# time the run
import timeit
start = timeit.default_timer()

# TODO: eventually use this format to write unit tests of each pdist method

# 2D Example: first initialize the h5 pdist class
#X, Y, Z = H5_Pdist(**data_options).run()
#plt.pcolormesh(X, Y, Z)

#X, Y, Z = H5_Plot(**data_options)
#H5_Plot(X, Y, Z).plot_hist_2d()

# TODO: I should be able to use the classes sepertely or together
#H5_Plot(plot_options=plot_options, **data_options).plot_contour()
wedap = H5_Plot(plot_options=plot_options, **data_options)
wedap.plot()
print(wedap.index_x)
#plt.savefig("west_c2.png")
#print(wedap.auxnames)

# TODO: test using different p_max values and bar

# 1D example
# pdist = H5_Pdist("data/west_i200.h5", aux_x="1_75_39_c2", **data_options)
# X, Y = pdist.run()
# plt.plot(X, Y)

stop = timeit.default_timer()
execution_time = stop - start
print(f"Executed in {execution_time:04f} seconds")

plt.show()

# TODO: build an tracer function that finds the iter,seg,frame(s) for an input hist bin