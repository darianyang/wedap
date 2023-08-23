"""
Cluster the data from west.h5 and get the iter,seg,frame for the cluster(s).

TODO: 
"""

import wedap
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

plt.style.use("styles/default.mplstyle")

# load h5 file into pdist class
data = wedap.H5_Pdist("wedap/data/p53.h5", data_type="average")

# extract weights
weights = data.get_all_weights()

# extract data arrays (can be pcoord or any aux data name)
X = data.get_total_data_array("pcoord", 0)
Y = data.get_total_data_array("pcoord", 1)

# put X and Y together column wise
XY = np.hstack((X,Y))

# scale data
scaler = StandardScaler()
XY = scaler.fit_transform(XY)

# -ln(W/W(max)) weights
weights_expanded = -np.log(weights/np.max(weights))

# cluster pdist using weighted k-means
clust = KMeans(n_clusters=5).fit(XY, sample_weight=weights_expanded)

# create plot base
fig, ax = plt.subplots()

# get color labels
cmap = np.array(["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", "#a65628"])
colors = [cmap[label] for label in clust.labels_.astype(int)]

# plot on PCs
pca = PCA(n_components=2)
PCs = pca.fit_transform(XY)
ax.scatter(PCs[:,0], PCs[:,1], c=colors, s=1)

# labels
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

plt.tight_layout()
plt.show()