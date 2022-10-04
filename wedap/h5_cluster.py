"""
Cluster the data from west.h5 and get the iter,seg,frame for the cluster(s).

TODO: 
"""

import wedap
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans
#from sklearn_extra.cluster import KMedoids
from sklearn.neighbors import KDTree

plt.style.use("styles/default.mplstyle")

pdist_options = {"h5" : "data/west_c2x_4b.h5",
                #"h5" : "data/skip_basis.h5",
                "Xname" : "1_75_39_c2",
                "Yname" : "rms_bb_nmr",
                #"Zname" : "rms_bb_nmr",
                "data_type" : "average",
                "weighted" : True,
                "p_units" : "kcal",
                "first_iter" : 400,
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
            #"plot_mode" : "hist2d",
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
Xo, Yo, Zo = data.pdist()

# turn array of arrays into 1D array column
# before this, they held value for each tau of each segment
#print("X pre: ", X.shape)
# X = Xo.reshape(-1,1)
# Y = Yo.reshape(-1,1)
#print(Y.shape)
#print("X post reshape: ", X.shape)

# old way of getting weight array
#weights = data.weights
#print("weights shape pre: ", weights.shape)
##weights = weights.reshape(-1,1)
#weights = np.concatenate(weights)
#print("weights shape post: ", weights.shape)
# need each weight value to be repeated for each tau (100 + 1) 
# to be same shape as X and Y
# weights_expanded = np.zeros(shape=(X.shape[0]))
# # loop over all ps intervals up to tau in each segment
# weight_index = 0
# for seg in weights:
#     for tau in range(0, Zo.shape[1]):
#         weights_expanded[weight_index] = seg
#         weight_index += 1

# now condensed into method
weights_expanded = data.get_all_weights()

# can use as basis for the method test
#np.savetxt("new_weights_expanded.txt", weights_expanded)

#print("new weight shape: ", weights_expanded.shape)

# can get the raw data arrays using this method now
X = data.get_total_data_array("auxdata/" + pdist_options["Xname"])
Y = data.get_total_data_array("auxdata/" + pdist_options["Yname"])

#np.testing.assert_array_equal(Y, Y2)

# put X and Y together column wise
XY = np.hstack((X,Y))

#weights_expanded = -np.log(weights_expanded)
#weights_expanded = 1 / weights_expanded
weights_expanded = -np.log(weights_expanded/np.max(weights_expanded))


from sklearn import cluster, mixture
n_clusters = 5

interval = 1000

# spectral clustering
#clust = cluster.SpectralClustering(n_clusters=n_clusters, eigen_solver="arpack", affinity="rbf").fit(XY[::interval,:])

# gmm clustering
clust = mixture.GaussianMixture(n_components=n_clusters, covariance_type="full").fit(XY[::interval,:])

# DBSCAN
#clust = cluster.DBSCAN().fit(XY[::interval,:])

# HDBSCAN
#import hdbscan
#clust = hdbscan.HDBSCAN().fit(XY[::interval,:])
#print(clust.labels_.max())
#clust.condensed_tree_.plot(select_clusters=True)
# import sys
# sys.exit()

# km cluster pdist
#clust = KMeans(n_clusters=n_clusters, random_state=0).fit(XY)
# can use weighted k-means but only clusters in high weight region (<10kT)
#clust = KMeans(n_clusters=n_clusters, random_state=0).fit(XY, sample_weight=weights_expanded)

if hasattr(clust, "labels_"):
    #cent = clust.cluster_centers_
    #print("Cluster Centers:\n", cent)
    #print("Sorted Cluster Centers:\n", np.sort(cent))
    labels = clust.labels_.astype(int)
    # plot km centers
    #plt.scatter(cent[:,0], cent[:,1], color="k")
else:
    labels = clust.predict(XY[::interval,:])
#labels = clust.labels_
#print("Cluster Labels:\n", labels)
#np.savetxt("test.txt", labels)

# make pdist plot
# plot = wedap.H5_Plot(plot_options=plot_options, plot_mode="hist2d", **plot_args, **pdist_options)
# plot.plot()

# plot the cluster labels for each point
# need a custom cmap
# import matplotlib.cm as cm
# from matplotlib.colors import Normalize
# cmap = cm.tab10
# norm = Normalize(vmin=0, vmax=10)     

# make colors array for each datapoint cluster label
#colors = [cmap(norm(label)) for label in labels] # too slow
cmap = np.array(["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
                 "#984ea3", "#999999", "#e41a1c", "#dede00", "#a65328"])
colors = [cmap[label] for label in labels]
                    
plot = wedap.H5_Plot(XY[::interval,0], XY[::interval,1], colors, plot_options=plot_options, cmap=cmap, plot_mode="scatter3d", **plot_args)
plot.plot(cbar=False)

plot.ax.set_title("GMM")
plt.tight_layout()

# TODO: filter by cluster


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
    for iter in range(data.first_iter, data.last_iter + 1):
        for seg in range(0, len(data.f[f"iterations/iter_{iter:08d}/seg_index"][:])):
            # shape of XYZ is originally (all n segs, per every n tau)
            for tau in range(0, Zo.shape[1]):
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

# find frame from WE closest to cluster center using kdtree query
# tree = KDTree(XY, leaf_size=10)
# # distances and indices of the k closest neighbors
# dist, ind = tree.query([cent[2]], k=3)
# print("DIST and INDX:\n", dist, ind)

# closest point to cluster of interest from tree query = ind[0,0]
# i, s, t = find_frame_from_index(ind[0,1])
# print(f"See ITER:{i}, SEG:{s}, FRAME:{t}")

stop = timeit.default_timer()
execution_time = stop - start
print(f"Executed in {execution_time:04f} seconds")

plt.show()

# these are the closest frames for the 35.82 3.60 cluster of i400-500 2KOD WE
# 404, 6, 80 | 487, 5, 35 |  433, 11, 20
# could look at the individual traced trajectories 
# (the latter 2 are more interesting)