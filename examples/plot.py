import wedap
import matplotlib.pyplot as plt

plt.style.use("styles/default.mplstyle")

data_options = {"h5" : "data/west_c2x.h5",
                #"h5" : "data/p53.h5",
                "Xname" : "1_75_39_c2",
                "Yname" : "rms_m2_xtal",
                #"Xname" : "pcoord",
                #"Yname" : "pcoord",
                #"Xindex" : 1,
                #"Yindex" : 0,
                #"Zname" : "rms_bb_nmr",
                "data_type" : "average",
                "p_max" : 30,
                "p_units" : "kcal",
                #"first_iter" : 490,
                #"last_iter" : 400, 
                #"bins" : 100,
                #"plot_mode" : "contour",
                #"cmap" : "gnuplot_r",
                "cbar_label" : "$-RT\ \ln\, P\ (kcal\ mol^{-1})$",
                #"data_smoothing_level" : 0.4,
                #"curve_smoothing_level" : 0.4,
                }

plot_options = {"ylabel" : "RMSD to Xtal ($\AA$)",
                "xlabel" : "Helical Angle (°)",
                "title" : "1A43 20-100° WE i490-500",
                "xlim" : (20,100),
                "ylim" : (1,7),
                "grid" : True,
                #"minima" : True, 
                }


plot = wedap.H5_Plot(plot_options=plot_options, **data_options)
plot.plot()

plt.show()

