"""
Eventually incorporate into tests.

TODO: figure out mpl styles and load a style for 1 col, 2 col, and poster figures.
"""
from h5_pdist import *
from h5_plot import *

data_options = {"h5" : "data/west_c2.h5",
                #"aux_x" : "1_75_39_c2",
                #"aux_y" : "angle_3pt",
                #"aux_x" : "rms_bb_nmr",
                "aux_x" : "M2_E175_chi1",
                "aux_y" : "M2_E175_chi2",
                "data_type" : "average",
                "p_max" : 5,
                "p_units" : "kcal",
                "first_iter" : 1,
                "last_iter" : 24,
                "bins" : 100,
                "bin_ext" : 0.01,
                #"plot_mode" : "line_1d",
                "plot_mode" : "hist_2d",
                #"data_smoothing_level" : 0.4,
                #"curve_smoothing_level" : 0.4,
                }
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                #"ylabel" : "RMSD ($\AA$)", 
                #"ylabel" : "WE Iteration", 
                #"xlabel" : "Helical Angle (°)",
                #"title" : "1A43 V02 100i WE",
                #"ylim" : (1, 4),
                #"xlim" : (20, 90),
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
H5_Plot(plot_options=plot_options, **data_options).plot()


# 1D example
# pdist = H5_Pdist("data/west_i200.h5", aux_x="1_75_39_c2", **data_options)
# X, Y = pdist.run()
# plt.plot(X, Y)

stop = timeit.default_timer()
execution_time = stop - start
print(f"Executed in {execution_time:04f} seconds")

plt.show()