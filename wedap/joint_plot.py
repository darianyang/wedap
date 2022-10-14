"""
Joint plot example using seaborn.
"""

import wedap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
                #"bins" : 100,
                }

plot_options = {"ylabel" : "RMSD to Xtal ($\AA$)",
                "xlabel" : "Helical Angle (°)",
                }

# instantiate wedap pdist class object
data = wedap.H5_Pdist("data/west_c2x_4b.h5", "average", first_iter=400)

# grab 1d weight per frame and scale
weights = data.get_all_weights()

# get the raw data arrays
X = data.get_total_data_array("1_75_39_c2")
Y = data.get_total_data_array("rms_bb_nmr")

# put X and Y together column wise
XY = np.hstack((X,Y))

# do some clustering
from sklearn.cluster import KMeans
clust = KMeans(n_clusters=4).fit(XY, sample_weight=weights)

# joint plot
df = pd.DataFrame(XY)
df["Cluster"] = clust.labels_
g = sns.jointplot(x=0, y=1, data=df.iloc[::100,:], 
                  kind="kde", hue="Cluster", palette="tab10")
g.set_axis_labels(**plot_options)

plt.show()