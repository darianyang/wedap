"""
Main call.
"""
from wekap.kinetics import *
from wekap.command_line import *
import pkgutil 
import os

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

    # if the list of direct h5 files is just one, carry on as normal
    if len(args.direct) == 1:
        # use first item in list of 1 item
        args.direct = args.direct[0]
        # vars converts from Namespace object to dict
        k = Kinetics(**vars(args))
        k.plot_rate()
    else:
        # use temp None but save original dh5 list
        multi_dh5 = args.direct
        # use temp as first item to get through init
        args.direct = args.direct[0]
        k = Kinetics(**vars(args))
        k.plot_multi_rates(multi_dh5)

    # plot formatting
    # take kwargs and unpack to look for plot option items
    k._unpack_plot_options()

    # option to plot exp D1D2 values
    # TODO: remove this since I can use avhline instead
    if args.exp_values:
        k.plot_exp_vals()

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