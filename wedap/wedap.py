"""
Main call.
"""

from command_line import *
from h5_pdist import *
from h5_plot import *

from search_aux import *
from h5_plot_trace import *

# if python file is being used 
if __name__ == '__main__': 
    
    # TODO: chicken and egg problem, when to initialize the parser?
    #f = h5py.File("data/west.h5", mode="r")
    #aux = list(f[f"iterations/iter_00000001/auxdata/"])

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
    # formatting, TODO: can include this in args
    plt.style.use("default.mplstyle")

    # TODO: clean this
    #H5_Plot(args, h5=args.h5, data_type=args.data_type).plot()

    if args.p_units == "kT":
        cbar_label = "$-\ln\,P(x)\ [kT^{-1}]$"
    elif args.p_units == "kcal":
        cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"

    # always output XYZ with fake Z for 1D, makes this easier
    X, Y, Z = H5_Pdist(args.h5, args.data_type, Xname=args.Xname, Yname=args.Yname, Zname=args.Zname, first_iter=args.first_iter, last_iter=args.last_iter, bins=args.bins, p_units=args.p_units).pdist()
    plot = H5_Plot(X, Y, Z, plot_mode=args.plot_mode, cmap=args.cmap, p_max=args.p_max, cbar_label=cbar_label)
    plot.plot()

    """
    Trace (Optional Argument) TODO: doesn not work with pcoord or evolution well
    """
    if args.trace_seg is not None:
        plot_trace(args.h5, args.trace_seg, args.Xname, args.Yname)
    if args.trace_val is not None:
        # for 1A43 V02: C2 and Dist M2-M1 - minima at val = 53° and 2.8A is alt minima = i173 s70
        # for demo: can use x = 53 and y = 2.7 or 2.6
        iter, seg = search_aux_xy_nn(args.h5, args.Xname, args.Yname, 
                                    # TODO: update to aux_x aux_y tuple
                                    args.trace_val[0], args.trace_val[1], args.last_iter)
        if args.data_type == "evolution": # TODO; this isn't the best
            evo = True
        else:
            evo = False
        plot_trace(args.h5, (iter,seg), args.Xname, args.Yname, ax=plot.ax, evolution=evo)

    """
    Plot formatting (TODO; handle multiple cli args here via plot_options?)
    """
    plot.ax.set_xlabel(args.Xname)
    if args.Yname:
        plot.ax.set_ylabel(args.Yname)
    if args.data_type == "evolution":
        plot.ax.set_ylabel("WE Iteration")

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
