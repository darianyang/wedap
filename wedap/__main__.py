"""
Main call.
"""

from matplotlib.pyplot import style
from wedap.h5_pdist import *
from wedap.h5_plot import *
from wedap.command_line import *

from wedap.search_aux import *
from wedap.h5_plot_trace import *

# for accessing package data: mpl styles
import pkgutil 
import os

def main():

    """
    Command line
    """
    # Create command line arguments with argparse
    argument_parser = create_cmd_arguments()
    # Retrieve list of args
    args = handle_command_line(argument_parser)

    """
    Generate pdist and plot it
    """
    if args.style == "default":
        # get the style parameters from package data
        # currently writes into temp file, could be more efficient (TODO)
        style = pkgutil.get_data(__name__, "styles/default.mplstyle")
        # pkgutil returns binary string, decode it first and make temp file
        with open("style.temp", "w+") as f:
            f.write(style.decode())
        plt.style.use("style.temp")
        # clean up temp style file
        os.remove("style.temp")
    elif args.style != "default" and args.style != "None":
        plt.style.use(args.style)

    # TODO: clean this
    #H5_Plot(args, h5=args.h5, data_type=args.data_type).plot()

    if args.p_units == "kT":
        cbar_label = "$-\ln\,P(x)$"
    elif args.p_units == "kcal":
        cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"

    # always output XYZ with fake Z for 1D, makes this part easier/less verbose
    X, Y, Z = H5_Pdist(args.h5, args.data_type, Xname=args.Xname, Xindex=args.Xindex, 
                       Yname=args.Yname, Yindex=args.Yindex, Zname=args.Zname, 
                       Zindex=args.Zindex, first_iter=args.first_iter, 
                       last_iter=args.last_iter, bins=args.bins, T=args.T,
                       weighted=args.weighted, p_units=args.p_units).pdist()
    plot = H5_Plot(X, Y, Z, plot_mode=args.plot_mode, cmap=args.cmap, 
                   p_max=args.p_max, cbar_label=cbar_label)#, color=args.color)
    plot.plot()

    """
    Trace (Optional Argument) TODO: does not work with pcoord or evolution well
    """
    if args.data_type == "evolution": # TODO; this isn't the best
        evo = True
    else:
        evo = False
    if args.trace_seg is not None:
        plot_trace(args.h5, args.trace_seg, args.Xname, args.Yname, ax=plot.ax, evolution=evo)
    if args.trace_val is not None:
        # for 1A43 V02: C2 and Dist M2-M1 - minima at val = 53deg and 2.8A is alt minima = i173 s70
        # for demo: can use x = 53 and y = 2.7 or 2.6
        print(args.h5, args.Xname, args.Yname, 
                                    # TODO: update to aux_x aux_y tuple
                                    args.trace_val[0], args.trace_val[1], args.last_iter)
        iter, seg = search_aux_xy_nn(args.h5, args.Xname, args.Yname, 
                                    # TODO: update to aux_x aux_y tuple
                                    args.trace_val[0], args.trace_val[1], 
                                    last_iter=args.last_iter)
        plot_trace(args.h5, (iter,seg), args.Xname, args.Yname, ax=plot.ax, evolution=evo)

    """
    Plot formatting (TODO; handle multiple cli args here via plot_options?)
    """
    plot.ax.set_xlabel(args.Xname)
    if args.Yname:
        plot.ax.set_ylabel(args.Yname)
    if args.data_type == "evolution":
        plot.ax.set_ylabel("WE Iteration")

    # args formatting (note args is a namespace object)
    if args.xlabel:
        plot.ax.set_xlabel(args.xlabel)
    if args.ylabel:
        plot.ax.set_ylabel(args.ylabel)
    if args.xlim:
        plot.ax.set_xlim(args.xlim)
    if args.ylim:
        plot.ax.set_ylim(args.ylim)
    if args.title:
        plot.ax.set_title(args.title)
    if args.cbar_label:
        plot.cbar.set_label(args.cbar_label, labelpad=14)

    """
    Show and/or save the final plot
    """
    plot.fig.tight_layout()
    # TODO: the save fig option produces a choppy image
    if args.output_path is not None:
        plot.fig.savefig(args.output_path, dpi=300, transparent=True)
    if args.output_to_screen is True:
        plt.show()
        #plot.fig.show() # only for after event loop starts e.g. with plt.show()

# if python file is being used 
if __name__ == "__main__": 
    main()