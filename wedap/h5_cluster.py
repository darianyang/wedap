"""
Cluster the data from west.h5 and get the iter,seg,frame for the cluster(s).

TODO: 
"""

import wedap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import KDTree

plt.style.use("styles/default.mplstyle")

pdist_options = {"h5" : "data/west_c2.h5",
                #"h5" : "data/skip_basis.h5",
                "Xname" : "1_75_39_c2",
                "Yname" : "rms_bb_xtal",
                #"Zname" : "rms_bb_nmr",
                "data_type" : "average",
                "weighted" : True,
                "p_units" : "kcal",
                "first_iter" : 400, # TODO: cant use with evolution (can use ylim)
                #"last_iter" : 425,
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
            #"p_max" : 50, # not working for 1D line plot (can use ylim)
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
data = wedap.H5_Pdist(**pdist_options, Zname="pcoord")
X, Y, Z = data.pdist()
weights = data.weights
#print("weights shape pre: ", weights.shape)
#weights = weights.reshape(-1,1)
weights = np.concatenate(weights)
#print("weights shape post: ", weights.shape)

# turn array of arrays into 1D array column
# before this, they held value for each tau of each segment
#print("X pre: ", X.shape)
X = X.reshape(-1,1)
Y = Y.reshape(-1,1)
#print("X post reshape: ", X.shape)

# need each weight value to be repeated for each tau (100 + 1) 
# to be same shape as X and Y
weights_expanded = np.zeros(shape=(X.shape[0]))
# loop over all ps intervals up to tau in each segment
weight_index = 0
for seg in weights:
    for tau in range(0, Z.shape[1]):
        weights_expanded[weight_index] = seg
        weight_index += 1

#print("new weight shape: ", weights_expanded.shape)

# put X and Y together column wise
XY = np.hstack((X,Y))

#weights_expanded = -np.log(weights_expanded)
#weights_expanded = 1 / weights_expanded

# cluster pdist
#km = KMeans(n_clusters=5, random_state=0).fit(XY)
# can use weighted k-means but only clusters in high weight region (<10kT)
km = KMeans(n_clusters=5, random_state=0).fit(XY, sample_weight=weights_expanded)
cent = km.cluster_centers_
print("Cluster Centers:\n", cent)
#print("Sorted Cluster Centers:\n", np.sort(cent))

labels = km.labels_
#print("Cluster Labels:\n", labels)
#np.savetxt("test.txt", labels)

# make pdist plot
plot = wedap.H5_Plot(plot_options=plot_options, **plot_args, **pdist_options)
plot.plot()

# plot km centers
plt.scatter(cent[:,0], cent[:,1], color="red")

# find frame from WE closest to cluster center using kdtree query
tree = KDTree(XY, leaf_size=10)
# distances and indices of the k closest neighbors
dist, ind = tree.query([cent[2]], k=3)
print("DIST and INDX:\n", dist, ind)

def find_frame_from_index(index):
    """
    Take index from querying a tree built from XY data with a cluster center
    and return an iteration, segment, and frame that corresponds to the 
    cluster center index.

    Parameters
    ----------
    index : int
        The index from kdtree query for a cluster center.
        This is the index of a 1D array with XY values at each tau value
        of all iterations and segments considered.
    
    Returns
    -------
    iter : int
    seg : int
    tau : int
    """
    we_index = 0
    found = False
    for iter in range(plot.first_iter, plot.last_iter + 1):
        for seg in range(0, len(plot.f[f"iterations/iter_{iter:08d}/seg_index"][:])):
            # shape of XYZ is originally (all n segs, per every n tau)
            for tau in range(0, Z.shape[1]):
                # check for the index of interest from tree
                if we_index == index:
                    #print(f"See ITER:{iter}, SEG:{seg}, FRAME:{tau}")
                    found = True
                    break
                # if not there yet, keep going until found
                we_index += 1
            if found:
                break
        if found:
            break

    return iter, seg, tau

# closest point to cluster of interest from tree query = ind[0,0]
i, s, t = find_frame_from_index(ind[0,1])
print(f"See ITER:{i}, SEG:{s}, FRAME:{t}")

stop = timeit.default_timer()
execution_time = stop - start
print(f"Executed in {execution_time:04f} seconds")

plt.show()

# these are the closest frames for the 35.82 3.60 cluster of i400-500 2KOD WE
# 404, 6, 80 | 487, 5, 35 |  433, 11, 20
# could look at the individual traced trajectories 
# (the latter 2 are more interesting)