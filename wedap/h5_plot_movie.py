
from turtle import width
from matplotlib.transforms import Bbox
import wedap
import matplotlib.pyplot as plt
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

plt.style.use("styles/default.mplstyle")

# iterations is about 100*duration with 20 fps and iteration = (t + 0.01) * 100
# seconds
duration = 6.5
#duration = 0.5
# frames per second
fps = 30

fig, axes = plt.subplots(1, 2, gridspec_kw={"width_ratios":[1,0.05]})


def make_frame(t):
    """
    Returns an image of the frame for time t.
    """
    axes[0].clear()
    axes[1].clear()

    #plt.clf()
    #plt.cla()

    #print(t)

    iteration = (t + 0.01) * 100
    #print(iteration)
          
    data_options = {"h5" : "data/west_c2x.h5",
                    #"h5" : "data/skip_basis.h5",
                    "Xname" : "1_75_39_c2",
                    #"Yname" : "angle_3pt",
                    "Yname" : "rms_bb_xtal",
                    #"Yname" : "rms_dimer_int_xtal",
                    #"Yname" : "rms_bb_xtal",
                    #"Xname" : "M1_E175_chi2",
                    #"Yname" : "M2_E175_chi2",
                    "data_type" : "average",
                    "p_max" : 30,
                    "p_units" : "kcal",
                    "first_iter" : int(iteration),
                    "last_iter" : int(iteration) + 100,
                    #"last_iter" : int(iteration),
                    "bins" : 100,
                    #"plot_mode" : "contour",
                    #"cmap" : "gnuplot_r",
                    "ax" : axes[0],
                    #"plot_mode" : "hist2d",
                    #"data_smoothing_level" : 0.4,
                    #"curve_smoothing_level" : 0.4,
                    }

    plot_options = {#"ylabel" : r"M2Oe-M1He1 Distance ($\AA$)", 
                    #"ylabel" : "RMSD to NMR ($\AA$)", 
                    "ylabel" : "RMSD to XTAL ($\AA$)", 
                    #"ylabel" : "WE Iteration", 
                    "xlabel" : "Helical Angle (°)",
                    #"title" : "2KOD C2 100i WE",
                    "title" : f"WE Iteration {int(iteration)} to {int(iteration) + 100}",
                    "ylim" : (1, 7),
                    #"xlim" : (10, 110),
                    "xlim" : (10, 120),
                    #"xlim" : (-180,180),
                    #"ylim" : (-180,180),
                    #"xlim" : (80,120),
                    "grid" : True,
                    #"minima" : True,
                    }
    
    we = wedap.H5_Plot(plot_options=plot_options, **data_options)
    # TODO: is this the best option? maybe it is, need to update
        # so if Yname, use cbar?
    we.plot(cbar=False)
    #we.cbar.remove()
    we.add_cbar(cax=axes[1])
    fig.tight_layout()

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=duration)
animation.write_gif('west_c2x.gif', fps=fps)