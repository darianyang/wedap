"""
Main call.
"""
import matplotlib.pyplot as plt

from mdap.md_pdist import *
from mdap.md_plot import *
from mdap.command_line import *

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

    # # for jointplots, save original p_units and run pdist with raw
    # og_p_units = args.p_units
    # #print(args.jointplot)
    # if args.jointplot is True:
    #     #vars(args)["p_units"] = "raw"
    #     args.p_units = "raw"

    # pdist = MD_Pdist(**vars(args))
    # X, Y, Z = pdist.pdist()

    # # TODO: update to MD_Plot for more flexibility and customization
    # # plot = MD_Plot(X, Y, Z, plot_mode=args.plot_mode, cmap=args.cmap,
    # #                contour_interval=args.contour_interval, p_min=args.p_min,
    # #                p_max=args.p_max, cbar_label=cbar_label, color=args.color,
    # #                smoothing_level=args.smoothing_level, jointplot=args.jointplot)
    # # plot = plot.plot()

    # # TODO: note that jointplot will not work well since p_units are not available
    # # for jointplot, use og p_units to convert from raw
    # if args.jointplot:
    #     args.p_units = og_p_units
    # plot = H5_Plot(X, Y, Z, **vars(args))
    # plot.plot()

    # MD_Plot use with implicit use of MD_Pdist
    plot = MD_Plot(**vars(args))
    plot.plot()

    # attempt to split to seperate pdist and plot classes
    # X, Y, Z  = MD_Pdist(**vars(args)).pdist()
    # plot = MD_Plot(**vars(args))
    # plot.plot()

    """
    Trace (Optional Argument) TODO: adapt to stdMD
    """
    # default to white if no color provided
    if args.color is None:
        args.color = "white"
    if args.trace_frame is not None: # TODO
        plot.plot_trace(args.trace_frame, color=args.color, ax=plot.ax)
    if args.trace_val is not None:
        iter, seg = plot.search_aux_xy_nn(args.trace_val[0], args.trace_val[1])
        plot.plot_trace((iter,seg), color=args.color, ax=plot.ax)

    """
    Plot formatting
    """
    # if cbar_label is given set as cbar_label, otherwise try to find a good label
    if args.cbar_label:
        cbar_label = args.cbar_label
    elif args.p_units == "kT":
        cbar_label = "$-\ln\,P(x)$"
    elif args.p_units == "kcal":
        cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"
    elif args.p_units == "raw":
        cbar_label = "Counts"
    elif args.p_units == "raw_norm":
        cbar_label = "Normalized Counts"
    # if using scatter3d and no label is given, create default label
    if (args.plot_mode == "scatter3d" or args.plot_mode == "hexbin3d") and args.cbar_label is None:
        cbar_label = plot.Zname[0] + " i" + str(plot.Zindex)
    # if there is a cbar object set label
    if hasattr(plot, "cbar"):
        plot.cbar.set_label(cbar_label, labelpad=14)
    # if using proj3d, add z axis label
    if args.proj3d is True:
        plot.ax.set_zlabel(cbar_label)

    # if no label is given, create default label (default to first item in XYZname list)
    if args.xlabel is None:
        # unless timeseries, then use time label
        if args.data_type == "time":
            plot.ax.set_xlabel(f"Time (frames x {str(args.timescale)})")
        else:
            plot.ax.set_xlabel(plot.Xname[0] + " i" + str(plot.Xindex))
    if args.ylabel is None:
        # use Xname on Y if timeseries, otherwise use Yname on Y
        if args.data_type == "time":
            plot.ax.set_ylabel(plot.Xname[0] + " i" + str(plot.Xindex))
        # if Y data is given
        elif args.Yname:
            plot.ax.set_ylabel(plot.Yname[0] + " i" + str(plot.Yindex))
        # otherwise it will be something like 1d pdist
        else:
            plot.ax.set_ylabel(cbar_label)

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