"""
Main call.
"""
from wedap.h5_pdist import *
from wedap.h5_plot import *
from wedap.command_line import *
from wedap.h5_gif import *

from wedap.logger import get_logger, set_log_level, format_call
from wedap.utils import apply_style

# use the package name (not __name__, which is "__main__" under `python -m wedap`)
logger = get_logger("wedap")

def _echo_call(args):
    """
    Build a copy-pasteable ``H5_Plot(...)`` string from the parsed args, keeping
    only the arguments that are actual H5_Plot/H5_Pdist constructor parameters.
    """
    import inspect
    params = set(inspect.signature(H5_Plot.__init__).parameters)
    params |= set(inspect.signature(H5_Pdist.__init__).parameters)
    params -= {"self", "args", "kwargs"}
    kwargs = {k: v for k, v in vars(args).items() if k in params and v is not None}
    return "wedap." + format_call("H5_Plot", kwargs)

def main():

    """
    Command line
    """
    # Create command line arguments with argparse
    argument_parser = create_cmd_arguments()
    # Retrieve list of args
    args = handle_command_line(argument_parser)

    # configure logging verbosity from CLI flags
    set_log_level(verbose=getattr(args, "verbose", False),
                  debug=getattr(args, "debug", False))

    """
    Generate pdist and plot it
    """
    # if not using screen use agg to be compatible with no display envs
    if args.no_output_to_screen:
        import matplotlib
        matplotlib.use('agg')

    # apply the requested matplotlib style (packaged default, named style, or path)
    apply_style(args.style, "wedap")

    # a poor workaround for now for the weighted arg
    if args.not_weighted is True:
        args.weighted = False
    elif args.not_weighted is False:
        args.weighted = True

    # default avg: when using scatter3d or hexbin3d, if Yname is provided (hence 2d and not evo),
    # or if requesting line plot, all these conditions only if dt is wrong (default evolution)
    if (args.plot_mode == "scatter3d" or args.plot_mode == "hexbin3d" or args.Yname is not None \
       or args.plot_mode == "line") and args.data_type == "evolution" :
        args.data_type = "average"

    #  make a gif instead of a single plot if gif_out is given
    # e.g. $ wedap -h5 wedap/data/p53.h5 -y pcoord -yi 1 -dt average --last-iter 16 
    #              --avg-plus 2 --gif-out test.gif --xlim 0 6 --ylim 0 36 --pmax 20
    if args.gif_out is not None:
        make_gif(**vars(args))
        # exit prematurely since gif making
        return
    # otherwise carry on as normal for a single plot
    else:
        # echo the resolved call so it can be lifted into a script/notebook
        logger.info(_echo_call(args))
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
            logger.error(f"{e}: Attempting to plot an {args.data_type} dataset using "
                         f"a {args.plot_mode} type plot. Is this what you meant to do?")
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
        iter, seg = plot.find_iter_seg_from_xy_vals(args.trace_val[0], args.trace_val[1])
        plot.plot_trace((iter,seg), color=args.color, ax=plot.ax, mark_points=args.mark_points)

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
        logger.info(f"Plot was saved to {args.output_path}")
        print(f"Plot was saved to {args.output_path}")
    if args.no_output_to_screen:
        pass
    else:
        plt.show()
        #plot.fig.show() # only for after event loop starts e.g. with plt.show()

# if python file is being used 
if __name__ == "__main__": 
    main()