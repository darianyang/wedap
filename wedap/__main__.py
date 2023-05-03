"""
Main call.
"""
from wedap.h5_pdist import *
from wedap.h5_plot import *
from wedap.command_line import *

#from wedap.search_aux import *
#from wedap.h5_plot_trace import *

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

    # if args.p_units == "kT":
    #     cbar_label = "$-\ln\,P(x)$"
    # elif args.p_units == "kcal":
    #     cbar_label = "$-RT\ \ln\, P\ (kcal\ mol^{-1})$"
    # elif args.p_units == "raw":
    #     cbar_label = "Counts"
    # elif args.p_units == "raw_norm":
    #     cbar_label = "Normalized Counts"

    # a poor workaround for now for the weighted arg
    # this is only to make the gooey formatting look nicer in terms of the checkbox
    if args.not_weighted is True:
        weighted = False
    elif args.not_weighted is False:
        weighted = True

    # accounting for joint plot option (TODO: better way?)
    # if args.jointplot:
    #     temp_p_units = "raw"
    # else:
    #     temp_p_units = args.p_units

    # always output XYZ with fake Z for 1D, makes this part easier/less verbose
    #pdist = H5_Pdist()
    # plot = H5_Plot(# H5_Pdist args
    #                 h5=args.h5, data_type=args.data_type, Xname=args.Xname,
    #                 Xindex=args.Xindex, Yname=args.Yname, Yindex=args.Yindex, Zname=args.Zname, 
    #                 Zindex=args.Zindex, first_iter=args.first_iter, skip_basis=args.skip_basis,
    #                 last_iter=args.last_iter, bins=args.bins, T=args.T,
    #                 weighted=weighted, p_units=args.p_units, no_pbar=args.no_pbar,
    #                 histrange_x=args.histrange_x, histrange_y=args.histrange_y,
    #                 # H5_Plot args
    #                 plot_mode=args.plot_mode, cmap=args.cmap,
    #                 contour_interval=args.contour_interval, p_min=args.p_min,
    #                 p_max=args.p_max, cbar_label=cbar_label, color=args.color,
    #                 smoothing_level=args.smoothing_level, jointplot=args.jointplot,
    #                 **args_dict)
    
    # vars converts from Namespace object to dict
    plot = H5_Plot(**vars(args))

    # had to adjust this since joint_plots require raw dist, so coupled pdist/plot needed
    # X, Y, Z = pdist.pdist()
    # plot = H5_Plot(X, Y, Z, plot_mode=args.plot_mode, cmap=args.cmap,
    #                contour_interval=args.contour_interval, p_min=args.p_min,
    #                p_max=args.p_max, cbar_label=cbar_label, color=args.color,
    #                smoothing_level=args.smoothing_level, jointplot=args.jointplot)
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
        if args.Yname:
            plot.ax.set_ylabel(args.Yname + " i" + str(plot.Yindex))
        if args.data_type == "evolution":
            plot.ax.set_ylabel("WE Iteration")

    # if cbar_label is given set as cbar_label
    if args.cbar_label:
        plot.cbar.set_label(args.cbar_label, labelpad=14)
    # if using scatter3d and no label is given, create default label
    elif args.plot_mode == "scatter3d":
        plot.cbar.set_label(args.Zname + " i" + str(plot.Zindex))

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