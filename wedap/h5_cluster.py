
import wedap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

plt.style.use("styles/default.mplstyle")

# TODO: auto ignore auxy when using 1d
pdist_options = {"h5" : "data/west_c2.h5",
                #"h5" : "data/skip_basis.h5",
                "Xname" : "1_75_39_c2",
                "Yname" : "rms_bb_xtal",
                #"Zname" : "rms_bb_nmr",
                "data_type" : "average",
                "weighted" : False,
                "p_units" : "kcal",
                "first_iter" : 400, # TODO: cant use with evolution (can use ylim)
                #"last_iter" : 750, 
                #"bins" : 100, # note bins affects contour quality
                }

plot_args = {#"plot_mode" : "contour",
            #"plot_mode" : "hexbin3d",
            #"cmap" : "gnuplot_r",
            #"cbar_label" : "RMSD ($\AA$)",
            "cbar_label" : "$-RT\ \ln\, P\ (kcal\ mol^{-1})$",
            #"cbar_label" : "$-\ln\,P(x)$",
            #"p_min" : 15,
            #"p_max" : 10, # not working for 1D line plot (can use ylim)
            "plot_mode" : "hist2d",
            #"plot_mode" : "scatter3d",
            #"plot_mode" : "line",
            }

# TODO: default to aux for labels if available or pcoord dim if None
plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                #"ylabel" : "RMSD ($\AA$)", 
                #"ylabel" : "WE Iteration", 
                "ylabel" : "RMSD to Xtal ($\AA$)",
                #"ylabel" : "$-RT\ \ln\, P\ (kcal\ mol^{-1})$", 
                "xlabel" : "Helical Angle (°)",
                #"ylabel" : "3 Point Angle (°)",
                "grid" : True,
                #"minima" : True, 
                # TODO: diagonal line option
                }


# time the run
import timeit
start = timeit.default_timer()

# generate raw data arrays
data = wedap.H5_Pdist(**pdist_options, Zname="rms_bb_nmr")
X, Y, Z = data.pdist()
weights = data.weights
#weights = weights.reshape(-1,1)
weights = np.concatenate(weights)
print("weights shape: ", weights.shape)

# turn array of arrays into 1D array column
# before this, they held value for each tau of each segment
print("X pre: ", X.shape)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
print("X post reshape: ", X.shape)

# need each weight value to be repeated for each tau (100 + 1) 
# to be same shape as X and Y
weights_expanded = np.zeros(shape=(X.shape[0]))
# loop over all ps intervals up to tau in each segment
weight_index = 0
for seg in weights:
    for tau in range(0, Z.shape[1]):
        weights_expanded[weight_index] = seg
        weight_index += 1

print("new weight shape: ", weights_expanded.shape)

# put X and Y together column wise
XY = np.hstack((X,Y))

# cluster pdist
km = KMeans(n_clusters=5, random_state=0).fit(XY)
# can use weighted k-means but only clusters in high weight region (<10kT)
#km = KMeans(n_clusters=5, random_state=0).fit(XY, sample_weight=weights_expanded)
#km = KMedoids(n_clusters=5, random_state=0).fit(XY)
cent = km.cluster_centers_
print(cent)

# make pdist plot
plot = wedap.H5_Plot(plot_options=plot_options, **plot_args, **pdist_options)
plot.plot()

# plot km centers
plt.scatter(cent[:,0], cent[:,1], color="red")

# TODO: next find frame from WE closest to cluster center using kdtree query

stop = timeit.default_timer()
execution_time = stop - start
print(f"Executed in {execution_time:04f} seconds")

plt.show()
