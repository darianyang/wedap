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
                "aux_y" : "1_75_39_c2",
                #"aux_y" : "angle_3pt",
                "aux_x" : "rog",
                #"aux_y" : "rms_dimer_int_xtal",
                #"aux_y" : "rms_bb_xtal",
                #"aux_x" : "M1_E175_chi2",
                #"aux_y" : "M2_E175_chi2",
                "data_type" : "average",
                "p_max" : 10,
                "p_units" : "kcal",
                #"first_iter" : 76,
                "last_iter" : 150,
                "bins" : 100,
                "plot_mode" : "hist2d",
                #"cmap" : "gnuplot_r",
                #"plot_mode" : "hist2d",
                #"data_smoothing_level" : 0.4,
                #"curve_smoothing_level" : 0.4,
                }

# TODO: default to aux for labels if available or pcoord dim if None
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                #"ylabel" : "RMSD ($\AA$)", 
                #"ylabel" : "WE Iteration", 
                #"xlabel" : "Helical Angle (°)",
                "title" : "2KOD C2 100i WE",
                #"ylim" : (2, 10),
                #"xlim" : (10, 110),
                #"xlim" : (30, 80),
                #"xlim" : (-180,180),
                #"ylim" : (-180,180),
                #"xlim" : (80,120),
                "grid" : True,
                #"minima" : True,
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