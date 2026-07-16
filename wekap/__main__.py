"""
Main call.
"""
from wekap.kinetics import *
from wekap.command_line import *

from wedap.logger import get_logger, set_log_level, format_call
from wedap.utils import apply_style

# use the package name (not __name__, which is "__main__" under `python -m wekap`)
logger = get_logger("wekap")

def _echo_call(args):
    """
    Build a copy-pasteable ``Kinetics(...)`` string from the parsed args, keeping
    only the arguments that are actual Kinetics constructor parameters.
    """
    import inspect
    params = set(inspect.signature(Kinetics.__init__).parameters) - {"self", "args", "kwargs"}
    kwargs = {k: v for k, v in vars(args).items() if k in params and v is not None}
    return "wekap." + format_call("Kinetics", kwargs)

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
    apply_style(args.style, "wekap")

    # warning if using default tau value
    if args.tau is None:
        args.tau = 100e-12
        logger.warning("Using the default tau value of 100 ps. "
                       "Set with the --tau option if this is not correct.")

    # if the list of direct h5 files is just one, carry on as normal
    if len(args.direct) == 1:
        # use first item in list of 1 item
        args.direct = args.direct[0]
        # echo the resolved Kinetics() call so it can be lifted into a script/notebook
        logger.info(_echo_call(args))
        # vars converts from Namespace object to dict
        k = Kinetics(**vars(args))
        k.plot_rate()
    else:
        # use temp None but save original dh5 list
        multi_dh5 = args.direct
        # use temp as first item to get through init
        args.direct = args.direct[0]
        logger.info(_echo_call(args))
        k = Kinetics(**vars(args))
        k.plot_multi_rates(multi_dh5)

    # plot formatting
    # take kwargs and unpack to look for plot option items
    k._unpack_plot_options()

    # option to plot reference values
    if args.ref_values is not None:
        k.plot_ref_vals(args.ref_values)

    # post process option before saving
    if args.postprocess_func is not None:
        k._run_postprocessing()

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