"""
Main call.
"""
from wedap.h5_pdist import *
from wedap.h5_plot import *
from wedap.command_line import *

# TODO: change to logging style instead of stdout
#import logging

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
    # if not using screen use agg to be compatible with no display envs
    if args.no_output_to_screen:
        import matplotlib
        matplotlib.use('agg')

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

    # a poor workaround for now for the weighted arg
    # this is only to make the gooey formatting look nicer in terms of the checkbox
    if args.not_weighted is True:
        args.weighted = False
    elif args.not_weighted is False:
        args.weighted = True
    
    # vars converts from Namespace object to dict
    plot = H5_Plot(**vars(args))

    # for 4d projected, adjust cbar position (this is a class attr)
    if args.proj4d is True:
        plot.cbar_pad = 0.2

    # 2D plot with cbar
    # TODO: can this be done better?
    if args.Yname or args.data_type == "evolution":
        try:
            plot.plot(cbar=True)
        except AttributeError as e:
            print(f"{e}: Attempting to plot an {args.data_type} dataset using ")
            print(f"a {args.plot_mode} type plot. Is this what you meant to do?")
            sys.exit(0)
    # 1D plot that isn't evolution
    elif args.Yname is None and args.Zname is None and args.data_type != "evolution":
        plot.plot(cbar=False)

    """
    Trace (Optional Argument)
    """
    # default to white if no color provided
    if args.color is None:
        args.color = "white"
    if args.trace_seg is not None:
        plot.plot_trace(args.trace_seg, color=args.color, ax=plot.ax)
    if args.trace_val is not None:
        iter, seg = plot.search_aux_xy_nn(args.trace_val[0], args.trace_val[1])
        plot.plot_trace((iter,seg), color=args.color, ax=plot.ax)

    """
    Plot formatting
    """
    # if no xlabel is given, create default label
    if args.xlabel is None:
        plot.ax.set_xlabel(args.Xname + " i" + str(plot.Xindex))

    # if no ylabel is given, create default label of Yname or "WE Iteration"
    if args.ylabel is None:
        if args.data_type == "evolution":
            plot.ax.set_ylabel("WE Iteration")
        if args.Yname:
            plot.ax.set_ylabel(args.Yname + " i" + str(plot.Yindex))

    # if cbar_label is given set as cbar_label
    if args.cbar_label:
        plot.cbar.set_label(args.cbar_label, labelpad=14)
    # if using scatter3d and no label is given, create default label
    elif args.plot_mode == "scatter3d" or args.plot_mode == "hexbin3d":
        # for 3d projected
        if args.proj3d is True:
            plot.ax.set_zlabel(args.Zname + " i" + str(plot.Zindex))
        # for 4d projected
        elif args.proj4d is True:
            plot.ax.set_zlabel(args.Zname + " i" + str(plot.Zindex))
            plot.cbar.set_label(args.Cname + " i" + str(plot.Cindex))
        # for 2d scatter with cbar
        else:
            plot.cbar.set_label(args.Zname + " i" + str(plot.Zindex))
    
    # run postprocessing function if requested
    if args.postprocess_func is not None:
        plot._run_postprocessing()

    """
    Show and/or save the final plot
    """
    # fig vs plt shouldn't matter here (needed to go plt for mosaic)
    #plot.fig.tight_layout()
    plt.tight_layout()
    if args.output_path is not None:
        # fig vs plt shouldn't matter here (needed to go plt for mosaic)
        #plot.fig.savefig(args.output_path)
        plt.savefig(args.output_path)
        #logging.info(f"Plot was saved to {args.output_path}")
        print(f"Plot was saved to {args.output_path}")
    if args.no_output_to_screen:
        pass
    else:
        plt.show()
        #plot.fig.show() # only for after event loop starts e.g. with plt.show()

# if python file is being used 
if __name__ == "__main__": 
    main()