"""
Functions for handling command-line input using argparse module.

TODO: 
     * add examples to desc
"""

import argparse
import sys

def create_cmd_arguments():
    """
    Use the `argparse` module to make the optional and required command-line
    arguments for the `wekap`. 

    Parameters 
    ----------

    Returns
    -------
    argparse.ArgumentParser: 
        An ArgumentParser that is used to retrieve command line arguments. 
    """
    wekap_desc = "================================================================ \n" + \
                 "=== Weighted Ensemble Kinetics Analysis and Plotting (WEKAP) === \n" + \
                 "================================================================ \n" + \
                 "\nPlot flux values from a direct.h5 file as rates"

    # create argument parser
    parser = argparse.ArgumentParser(description=wekap_desc,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    ###########################################################
    ######################## ARGUMENTS ########################
    ###########################################################
    # nargs = '?' "One argument will be consumed from the command line if possible, 
        # and produced as a single item. If no command-line argument is present, 
        # the value from default will be produced."

    main = parser.add_argument_group("Main Arguments")
    optional = parser.add_argument_group("Optional Extra Arguments")

    # TODO: allow for multiple h5 files and calc error using bootstrapping
    main.add_argument(
        "--direct", "--direct-h5", "-dh5",
        dest="direct",
        type=str,
        default="direct.h5",
        nargs="*", # optional multi args
        help="Name of output direct.h5 file from WESTPA w_direct or w_ipa.",
    )
    main.add_argument(
        "--assign", "--assign-h5", "-ah5",
        dest="assign",
        type=str,
        default=None,
        #nargs="*", # optional multi args (TODO: account for multi assign files as well)
        help="Name of specific assign.h5 file. Needed for labeled population data. "
             "But only if your `statepop` choice is `assign`. By default will use "
             "state populations from `direct.h5`.",
    )
    main.add_argument(
        "--tau", "-t",
        dest="tau",
        type=float,
        default=None,
        help="The resampling interval of the WE simulation in seconds. "
             "Default: 100ps = 100 * 10^-12 (s).",
    )
    main.add_argument(
        "--state",
        dest="state",
        type=int,
        default=1,
        help="State for flux calculations (flux into `state`), 0 = A and 1 = B, etc. "
             "By default, state = 1 (flux into state B)."
    )
    main.add_argument(
        "-c", "--concentration", "--conc",
        dest="concentration",
        type=float,
        default=1,
        help="Concentration (in Molar) to normalize the rate by. Default 1 (no-op). "
             "Divide the rate by this value, e.g. for a bimolecular (2nd order) rate "
             "constant from a pseudo-first-order WE simulation."
    )
    main.add_argument("-o", "--output", default=None,
                        dest="output_path",
                        help="The filename to which the plot will be saved. "
                             "Various image formats are available. You " 
                             "may choose one by specifying an extension. "
                             "\nLeave this empty if you don't want to save "
                             "the plot to a serparate file.",
                        type=str)
    optional.add_argument(
        "--label",
        dest="label",
        type=str,
        default=None,
        help="Data label."
    )
    optional.add_argument(
        "--statepop",
        dest="statepop",
        type=str,
        default="direct",
        choices=["direct", "assign"],
        help="'direct' for state_population_evolution from direct.h5 or 'assign' for "
             "labeled_populations from assign.h5. By default will use populations from "
             "direct.h5, note that if you use the populations from assign.h5 instead, "
             "they will need to be consistent with any window or cumulative averaging schemes "
             "used for direct.h5 rate calculations."
    )
    optional.add_argument(
        "-fu", "--flux-units",
        dest="flux_units",
        type=str,
        default="rates",
        choices=["rates", "mfpts"],
        help="Flux measurement units. Can be `rates` (default) or `mfpts`."
    )
    optional.add_argument(
        "--savefig",
        dest="savefig",
        type=str,
        default=None,
        help="Path to optionally save the figure."
    )
    # TODO: could expand this option better
    optional.add_argument(
        "--ref-values", "-ref",
        dest="ref_values",
        type=float,
        default=None,
        help="Plot reference (e.g. experimental) rate value as a black dashed horizontal line."
    )
    optional.add_argument(
        "--no-cumulative-avg", "-ncavg", "-nca",
        dest="cumulative_avg",
        action="store_false",
        help="Set when kinetics were NOT calculated with cumulative averaging. "
             "E.g. if you have instantaneous flux. This is only when using the "
             "assign.h5 file for state populations (--statepop assign)."
    )
    optional.add_argument("-xu", "--x-units", "--xunits",
                        dest = "x_units", type=str, default="iterations",
                        choices=["iterations", "moltime", "agg"],
                        help = "X units can be `iteration`, `moltime`, or `agg`.") 
    optional.add_argument("-nots", "--no-output-to-screen",
                        dest = "no_output_to_screen",
                        help = "Include this argument to not output the plot to "
                        "your display.", 
                        action= "store_true") 
    optional.add_argument("-red", "--red",
                        dest = "red",
                        help = "Optionally correct the rate using the Rate from Event "
                               "Durations (RED) scheme, computed from the `durations` "
                               "dataset in the direct.h5 file.",
                        action= "store_true")
    optional.add_argument("-rtp", "--red-timepoints",
                        dest = "red_timepoints",
                        type=int, default=None,
                        help = "Number of timepoints (pcoord frames) per iteration for "
                               "the RED correction resolution, including first and last "
                               "(e.g. 21 for a pcoord saved every 1 ps with tau=20 ps). "
                               "Default auto-detects from assign.h5 `npts`, else uses 2.")
    optional.add_argument("-v", "--verbose",
                        dest = "verbose",
                        help = "Enable verbose (INFO level) logging output.",
                        action= "store_true")
    optional.add_argument("--debug",
                        dest = "debug",
                        help = "Enable debug level logging output.",
                        action= "store_true")

    ##########################################################
    ############### FORMATTING ARGUMENTS #####################
    ##########################################################

    formatting = parser.add_argument_group("Plot Formatting Arguments") 

    formatting.add_argument("--style", default="default", nargs="?",
                        dest="style",
                        help="mpl style, can leave blank to use default, "
                             "input `None` for basic mpl settings, can use a custom "
                             "path to a mpl.style text file, or could use a mpl included "
                             "named style, e.g. `ggplot`. "
                             "Edit the wekap/styles/default.mplstyle file to "
                             "change default wekap plotting style options.",
                        type=str)
    formatting.add_argument("--color",
                        dest="color", help="Color of lines.")
    formatting.add_argument("--linewidth", "-lw", default=None, nargs="?",
                        dest="linewidth", help="Linewidth thickness.",
                        type=float)
    formatting.add_argument("--linestyle", "-ls", default="-", nargs="?",
                        dest="linestyle", help="Linestyle.",
                        type=str)
    formatting.add_argument("--xlabel", dest="xlabel", type=str)
    formatting.add_argument("--xlim", help="LB UB", dest="xlim", nargs=2, type=float)
    formatting.add_argument("--ylabel", dest="ylabel", type=str)
    formatting.add_argument("--ylim", help="LB UB", dest="ylim", nargs=2, type=float)
    formatting.add_argument("--title", dest="title", type=str)
    formatting.add_argument("--suptitle", dest="suptitle", type=str)
    formatting.add_argument("--grid", dest="grid", default=False, action="store_true")
    formatting.add_argument("--axvline", "-vl", help="Can be a single value or a list of lines.", 
                            dest="axvline", nargs="*", type=float)
    formatting.add_argument("--axhline", "-hl", help="Can be a single value or a list of lines.",
                            dest="axhline", nargs="*", type=float)
#     formatting.add_argument("--text", help="3 args for ax.text: x, y, string",
#                             dest="text", nargs="3", type=float)
#     formatting.add_argument("--figsize", default=None, nargs=2,
#                             dest="figsize",
#                             help="Matplotlib figure size, e.g. (6,4)",
#                             type=float)
    formatting.add_argument('--postprocess', '--postprocess-function', '-ppf', default=None,
                            dest='postprocess_func',
                            help='After plotting data, load and execute the '
                                 'Python function specified by '
                                 'POSTPROCESS_FUNC. POSTPROCESS_FUNC should be '
                                 'a string of the form ``mymodule.myfunction``.'
                                 'Example: '
                                 '``--postprocess mymodule.myfunction``.')

    # return the argument parser
    return parser 

# TODO: adjust all to fix str/int/type auto
def handle_command_line(argument_parser): 
    """
    Take command line arguments, check for issues, return the arguments. 

    Parameters
    ----------
    argument_parser : argparse.ArgumentParser 
        The argument parser that is returned in `create_cmd_arguments()`.
    
    Returns
    -------
    argparse.NameSpace
        Contains all arguments passed into wedap.
    
    Raises
    ------  
    Prints specific issues to terminal.
    """
    # retrieve args
    args = argument_parser.parse_args() 

     # TODO: check make sure that -jp and -3d not being using at same time

    # h5 file and file exists
    # if not os.path.exists(args.file) or not ".h5" in args.file:  
    #     # print error message and exits
    #     sys.exit("Must input file that exists and is in .h5 file format.")

    # if not args.percentage.isdigit():  # not correct input   
    #     # print out any possible issues
    #     print("You must input a percentage digit. EXAMPLES: \
    #     \n '-p 40' \n You CANNOT add percent sign (eg. 50%) \n You \
    #     CANNOT add decimals (eg. 13.23)") 
    #     sys.exit(0) # exit program

    # # ignore if NoneType since user doesn't want --maxEnsembleSize parameter
    # if args.max_ensemble_size is None: 
    #     pass

    # elif not args.max_ensemble_size.isdigit(): # incorrect input 
    #     # needs to be whole number
    #     print("You must input a whole number with no special characters (eg. 4).")  
    #     sys.exit(0) # exit program 

    # elif args.max_ensemble_size is '0': # user gives 0 to --maxEnsembleSize flag 
    #     print("You cannot input '0' to --maxEnsembleSize flag.")
    #     sys.exit(0) # exit program 

    return args