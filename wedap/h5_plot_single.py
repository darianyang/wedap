"""
Eventually incorporate into tests.

TODO: figure out mpl styles and load a style for 1 col, 2 col, and poster figures.
"""
from h5_pdist import *
from h5_plot import *

data_options = {"data_type" : "average",
                "p_max" : 50,
                "p_units" : "kcal",
                "first_iter" : 100,
                "last_iter" : 200,
                "bins" : 100
                }
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                "ylabel" : "M2 RMSD ($\AA$)", 
                "xlabel" : "Helical Angle (°)",
                #"title" : "1A43 V02 100i WE",
                #"ylim" : (0, 20),
                #"xlim" : (0, 90),
                "grid" : True,
                #"minima" : True,
                #"xlim" : (2,8)
                }


# TODO: eventually use this format to write unit tests of each pdist method

# 2D Example: first initialize the h5 pdist class
X, Y, Z = H5_Pdist("data/west_i200.h5", aux_x="1_75_39_c2", aux_y="fit_m1_rms_heavy_m2", **data_options).run()
plt.pcolormesh(X, Y, Z)
plt.colorbar()


# 1D example
# pdist = H5_Pdist("data/west_i200.h5", aux_x="1_75_39_c2", **data_options)
# X, Y = pdist.run()
# plt.plot(X, Y)


plt.show()