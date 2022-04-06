
from west_h5_plotting import *

data_options = {"data_type" : "instant",
                #"p_max" : 20,
                "p_units" : "kcal",
                "last_iter" : 100,
                "bins" : 100
                }
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                "ylabel" : "M2 RMSD ($\AA$)", 
                "xlabel" : "Helical Angle (°)",
                #"title" : "1A43 V02 100i WE",
                "ylim" : (0, 20),
                "xlim" : (0, 90),
                "grid" : True,
                #"minima" : True,
                #"xlim" : (2,8)
                }

# X, Y, Z = pdist_to_normhist("data/west_i200_crawled.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
# plot_normhist(X, Y, Z, plot_type="contour", cmap="gnuplot_r", **data_options, **plot_options)

# initialize the h5 plotting class
plotter = West_H5_Plotting("data/west_i200.h5", aux_x="1_75_39_c2", aux_y="fit_m1_rms_heavy_m2", **data_options)

# run pdist method
X, Y, Z = plotter.pdist_to_normhist()
# run plot method
plotter.plot_normhist(X, Y, norm_hist=Z, **plot_options)

plt.show()