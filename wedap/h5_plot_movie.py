
from h5_plot import *
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

plt.style.use("default.mplstyle")

# iterations is about 100*duration with 20 fps and iteration = (t + 0.01) * 100
# seconds
duration = 1.5
# frames per second
fps = 20

fig, ax = plt.subplots()
def make_frame(t):
    """
    Returns an image of the frame for time t.
    """
    ax.clear()

    #print(t)

    iteration = (t + 0.01) * 100

          
    data_options = {"h5" : "data/west_c2.h5",
                    "aux_x" : "1_75_39_c2",
                    #"aux_y" : "angle_3pt",
                    "aux_y" : "rms_heavy_xtal",
                    #"aux_y" : "rms_dimer_int_xtal",
                    #"aux_y" : "rms_bb_xtal",
                    #"aux_x" : "M1_E175_chi2",
                    #"aux_y" : "M2_E175_chi2",
                    "data_type" : "average",
                    "p_max" : 15,
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
                    #"ylabel" : "WE Iteration", 
                    #"xlabel" : "Helical Angle (°)",
                    #"title" : "2KOD C2 100i WE",
                    "title" : f"WE Iteration {int(iteration)}",
                    "ylim" : (2, 8),
                    #"xlim" : (10, 110),
                    "xlim" : (30, 85),
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