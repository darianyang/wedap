
from h5_plot_main import *

data_options = {"data_type" : "average",
                "p_max" : 20,
                "p_units" : "kcal",
                "last_iter" : 100,
                "bins" : 100
                }
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                "ylabel" : r"M2 RMSD ($\AA$)", 
                "xlabel" : "Helical Angle (°)",
                "title" : "1A43 V02 100i WE",
                "ylim" : (0, 20),
                "xlim" : (0, 90),
                "grid" : True,
                "minima" : True,
                #"xlim" : (2,8)
                }

# X, Y, Z = pdisZ,t_to_normhist("2kod_v02/wcrawl/west_i200_crawled.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
# plot_normhist(X, Y,  plot_type="contour", cmap="gnuplot_r", **data_options, **plot_options)

# X, Y, Z = pdist_to_normhist("2kod_v03/v01/west_i200.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
# plot_normhist(X, Y, Z, plot_type="contour", cmap="gnuplot_r", **data_options, **plot_options)

# X, Y = pdist_to_normhist("2kod_v02/wcrawl/west_i200_crawled.h5", "1_75_39_c2", **data_options)
# plot_normhist(X, Y, **data_options, **plot_options)

# X, Y, Z = pdist_to_normhist("1a43_v01/wcrawl/west_i150_crawled.h5", "1_75_39_c2", "M2Oe_M1He1", **data_options)
# plot_normhist(X, Y, Z, plot_type="heat", cmap="gnuplot_r", **data_options, **plot_options)

# X, Y, Z = pdist_to_normhist("1a43_v02/wcrawl/west_i200_crawled.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
# plot_normhist(X, Y, Z, plot_type="contour", cmap="gnuplot_r", **data_options, **plot_options)

X, Y, Z = pdist_to_normhist("1a43_v03/v01/west_i200.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
plot_normhist(X, Y, Z, plot_type="contour", cmap="gnuplot_r", **data_options, **plot_options)
