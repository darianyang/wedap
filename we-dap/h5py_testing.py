import h5py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### exploration of h5 file ###
#f = h5py.File("west_133i.h5", mode='r')
# total = 0

#string = "iterations/iter_" + str(i).zfill(8) + "/seg_index"
#string = "summary"

# parse h5 file to find auxillary data
#hdf_str_data = f[string]
#pd.DataFrame(hdf_str_data).to_csv(f"test.tsv", sep='\t')
#print(pd.DataFrame(hdf_str_data))
#print(list(hdf_str_data))
#print(np.ndarray(hdf_str_data))


#segs = hdf_str_data.shape[0]
#print(f"for iter {i}, there are {segs} segs.")
#total += segs
#print(hdf_str_data.shape[1])

# aux_data = (len(trace[0]) - 1) *   

# target should be 13433 rows / datapoints (101 pcoord points * 133 iter)
#print("rows = ", total)


### exploration of h5 file ###
#f = h5py.File("2kod_v02/pdist_i200_rmsheavy_rog.h5", mode='r')
# f = h5py.File("2kod_v02/west_i200.h5", mode='r')
# total = 0

# iter = 1
# #string = f"iterations/iter_{iter:08d}/seg_index"
# #string = "histograms"
# print(np.shape(pd.DataFrame(f[f"iterations/iter_{iter:08d}/auxdata/RoG"]).to_numpy()))
# print(np.shape(pd.DataFrame(np.array(f[f"iterations/iter_{iter:08d}/auxdata/RoG"]))))
# print(np.shape(np.array(f[f"iterations/iter_{iter:08d}/auxdata/RoG"])))

#parse h5 file to find auxillary data
# hdf_str_data = f[string]
# print(pd.DataFrame(hdf_str_data[199]))
# pd.DataFrame(hdf_str_data[199]).to_csv(f"pdist_hist_test_2.tsv", sep='\t')
#print(np.loadtxt(hdf_str_data))


### more exploration ###
# f = h5py.File("2kod_v02/west_i133.h5", mode="r")
# weights = f[f"iterations/iter_00000133/seg_index"]
# print(f.attrs["west_current_iteration"] - 1)
# #df = pd.DataFrame(weights).to_numpy()
# df = np.array(weights)
# print(df[0][0])

#df.to_csv("i187_segindex.tsv", sep="\t")
# array = df.to_numpy()
# weights = [array[i][0][0] for i in range(0,115)]
# plt.plot(weights)
# plt.show()


def plot_weights(h5, iter):
    f = h5py.File(h5, mode="r")
    weights = np.array(f[f"iterations/iter_{iter:08d}/seg_index"])
    weights = [weights[i][0] for i in range(0,weights.shape[0])]

    max_wlk = np.where(weights == np.max(weights))
    min_wlk = np.where(weights == np.min(weights))
    print(f"Max weight walker = {max_wlk} \n Min weight walker = {min_wlk}")

    plt.plot(weights)
    plt.title(f"Iteration {iter}")
    plt.show()

#plot_weights("1a43_v02/wcrawl/west_i200_crawled.h5", 200)

iteration = 1
f = h5py.File("data/west.h5", mode="r")
aux = list(f[f"iterations/iter_{iteration:08d}/auxdata/"])
print(aux)