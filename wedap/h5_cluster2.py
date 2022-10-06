"""
Testing streamlined clustering helper methods.
using p53.h5 as a simpler test dataset.
"""

import wedap
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.cluster import KMeans
from sklearn import cluster, mixture

plt.style.use("styles/default.mplstyle")

# generate raw data arrays
data = wedap.H5_Pdist("data/p53.h5", data_type="average", last_iter=15)

# now condensed into method
weights_expanded = data.get_all_weights()

# can get the raw data arrays using this method now
X = data.get_total_data_array("pcoord", 0)
Y = data.get_total_data_array("pcoord", 1)

print(X)

# goal is to now load it in and reshape it
np.savetxt("p53_X_array_i15.txt", X)

sys.exit(0)

# put X and Y together column wise
XY = np.hstack((X,Y))

weights_expanded = -np.log(weights_expanded/np.max(weights_expanded))

n_clusters = 5
interval = 10

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

if hasattr(clust, "labels_"):
    #cent = clust.cluster_centers_
    #print("Cluster Centers:\n", cent)
    #print("Sorted Cluster Centers:\n", np.sort(cent))
    labels = clust.labels_.astype(int)
    # plot km centers
    #plt.scatter(cent[:,0], cent[:,1], color="k")
else:
    labels = clust.predict(XY[::interval,:])

cmap = np.array(["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628",
                 "#984ea3", "#999999", "#e41a1c", "#dede00", "#a65328"])
colors = [cmap[label] for label in labels]
                    
plot = wedap.H5_Plot(XY[::interval,0], XY[::interval,1], colors, cmap=cmap, plot_mode="scatter3d",)
plot.plot(cbar=False)

plot.ax.set_title("GMM")
plt.tight_layout()
plt.show()
