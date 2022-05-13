"""
Plot a traced WE trajectory onto 2D plots.
# TODO: integrate into h5_plot
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py


def get_parents(walker_tuple, h5_file):
    it, wlk = walker_tuple
    parent = h5_file[f"iterations/iter_{it:08d}"]["seg_index"]["parent_id"][wlk]
    return it-1, parent

def trace_walker(walker_tuple, h5_file):
    # Unroll the tuple into iteration/walker 
    it, wlk = walker_tuple
    # Initialize our path
    path = [(it,wlk)]
    # And trace it
    while it > 1: 
        it, wlk = get_parents((it, wlk), h5_file)
        path.append((it,wlk))
    return np.array(sorted(path, key=lambda x: x[0]))

def get_aux(path, h5_file, aux_name):
    # Initialize a list for the pcoords
    aux_coords = []
    # Loop over the path and get the pcoords for each walker
    for it, wlk in path:
        # Here we are taking every 10 time points, feel free to adjust to see what that does
        aux_coords.append(h5_file[f'iterations/iter_{it:08d}/auxdata/{str(aux_name)}'][wlk][::10])
        #pcoords.append(h5_file[f'iterations/iter_{it:08d}']['pcoord'][wlk][::10,:])
    return np.array(aux_coords)

def plot_trace(h5, walker_tuple, aux_x, aux_y=None, evolution=False, ax=None):
    """
    Plot trace.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,5))
    else:
        fig = plt.gcf()

    it, wlk = walker_tuple
    with h5py.File(h5, "r") as w:
        # adjustments for plothist evolution of only aux_x data
        if evolution:
            # split iterations up to provide y-values for each x-value (pcoord)
            iter_split = [i + (j/aux_x.shape[1]) 
                          for i in range(0, it) 
                          for j in range(0, aux_x.shape[1])]
            ax.plot(aux_x[:,0], iter_split, c="black", lw=2)
            ax.plot(aux_x[:,0], iter_split, c="white", lw=1)
            return

        path = trace_walker((it, wlk), w)

        # And pull aux_coords for the path calculated
        aux_x = get_aux(path, w, aux_x)
        aux_y = get_aux(path, w, aux_y)

        ax.plot(aux_x[:,0], aux_y[:,0], c="black", lw=2)
        ax.plot(aux_x[:,0], aux_y[:,0], c="cyan", lw=1)


# from h5_plot_main import *
# data_options = {"data_type" : "average",
#                 "p_max" : 20,
#                 "p_units" : "kcal",
#                 "last_iter" : 200,
#                 "bins" : 100
#                 }

# h5 = "1a43_v02/wcrawl/west_i200_crawled.h5"
# aux_x = "1_75_39_c2"
# aux_y = "M2Oe_M1He1"

# X, Y, Z = pdist_to_normhist(h5, aux_x, aux_y, **data_options)
# levels = np.arange(0, data_options["p_max"] + 1, 1)
# plt.contour(X, Y, Z, levels=levels, colors="black", linewidths=1)
# plt.contourf(X, Y, Z, levels=levels, cmap="gnuplot_r")
# plt.colorbar()

# from search_aux import *
# # for 1A43 V02: C2 and Dist M2-M1 - minima at val = 53° and 2.8A is alt minima = i173 s70
# iter, seg = search_aux_xy_nn(h5, aux_x, aux_y, 53, 2.8, data_options["last_iter"])
# plot_trace(h5, (iter,seg), aux_x, aux_y)

# plt.show()
