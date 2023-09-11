
import wedap
import matplotlib.pyplot as plt

with plt.xkcd():
    plot = wedap.H5_Plot(h5="p53.h5", data_type="evolution")
    plot.plot()
    iter, seg = plot.search_aux_xy_nn(6, 17.5)
    plot.plot_trace((iter, seg), ax=plot.ax)
    plot.cbar.set_label("-ln P(x)")
    plt.xlabel("Progress Coordinate")
    plt.ylabel("WE Iteration")
    plt.tight_layout()
    plt.savefig("xkcd.pdf", transparent=True)
