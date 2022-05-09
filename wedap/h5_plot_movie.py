
from h5_plot import *
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

plt.style.use("default.mplstyle")

# iterations is about 100*duration with 20 fps and iteration = (t + 0.01) * 100
# seconds
duration = 4.0
# frames per second
fps = 30

fig, ax = plt.subplots()
def make_frame(t):
    """
    Returns an image of the frame for time t.
    """
    ax.clear()
    #plt.clf()

    #print(t)

    iteration = (t + 0.01) * 100

          
    data_options = {"h5" : "data/west_c2.h5",
                    "Xname" : "1_75_39_c2",
                    #"Yname" : "angle_3pt",
                    "Yname" : "rms_bb_xtal",
                    #"Yname" : "rms_dimer_int_xtal",
                    #"Yname" : "rms_bb_xtal",
                    #"Xname" : "M1_E175_chi2",
                    #"Yname" : "M2_E175_chi2",
                    "data_type" : "instant",
                    "p_max" : 30,
                    "p_units" : "kcal",
                    #"first_iter" : 100,
                    "last_iter" : int(iteration),
                    "bins" : 100,
                    "plot_mode" : "contour",
                    "cmap" : "gnuplot_r",
                    "ax" : ax,
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
                    "title" : f"WE Iteration {int(iteration)}",
                    "ylim" : (2, 8),
                    #"xlim" : (10, 110),
                    "xlim" : (20, 100),
                    #"xlim" : (-180,180),
                    #"ylim" : (-180,180),
                    #"xlim" : (80,120),
                    "grid" : True,
                    #"minima" : True,
                    }

    wedap = H5_Plot(plot_options=plot_options, **data_options)
    wedap.plot()
    wedap.cbar.remove()

    return mplfig_to_npimage(fig)

animation = VideoClip(make_frame, duration=duration)
animation.write_gif('west_c2.gif', fps=fps)