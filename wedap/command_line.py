"""
Functions for handling command-line input using argparse module.

TODO: add option to split pdist and plot once the pdist to txt feature is done
      so the pdist txt file could be: X_column, Y_column, Z_matrix_columns.
      This would be something like pdist.dap default and option to check and 
      read in a wedap or westpa pdist file for pdist function.
"""

import argparse
import sys

# import and use gooey conditionally
# adapted from https://github.com/chriskiehl/Gooey/issues/296
try:
    import gooey
    #from gooey import Gooey
    #from gooey import GooeyParser
except ImportError:
    gooey = None

def flex_add_argument(f):
    """Make the add_argument accept (and ignore) the widget option."""

    def f_decorated(*args, **kwargs):
        kwargs.pop('widget', None)
        return f(*args, **kwargs)

    return f_decorated

# Monkey-patching a private class…
argparse._ActionsContainer.add_argument = \
    flex_add_argument(argparse.ArgumentParser.add_argument)

# Do not run GUI if it is not available or if command-line arguments are given.
if gooey is None or len(sys.argv) > 1:
    ArgumentParser = argparse.ArgumentParser

    def gui_decorator(f):
        return f
else:
    ArgumentParser = gooey.GooeyParser
    gui_decorator = gooey.Gooey(
        program_name='wedap',
        #navigation='TABBED',
        #advanced=True,
        suppress_gooey_flag=True,
        optional_cols=4, 
        default_size=(1000, 600),
        #tabbed_groups=True,
    )

# TODO: make tabs?
@gui_decorator
def create_cmd_arguments(): 
    """
    Use the `argparse` module to make the optional and required command-line
    arguments for the `wedap`. 

    Parameters 
    ----------

    Returns
    -------
    argparse.ArgumentParser: 
        An ArgumentParser that is used to retrieve command line arguments. 
    """
    wedap_desc = "============================================================ \n" + \
                 "=== weighted ensemble data analysis and plotting (wedap) === \n" + \
                 "============================================================ \n" + \
                 "\nGiven an input west.h5 file from a successful WESTPA simulation, " + \
                 "prepare probability distributions and plots." + \
                 "\nSee the documentation for usage and examples: https://darianyang.github.io/wedap" + \
                 "\n\n" + \
                 "wedap can be used with 3 different --data-type (-dt) args: " + \
                 "\n\t`evolution` (default), `average`, and `instant`" + \
                 "\n\nAvailable --plot-mode (-pm) options are: " + \
                 "\n\t1D: `line`" + \
                 "\n\t2D: `hist` (default), `hist_l` (hist with contour lines), " + \
                 "\n\t    `contour` (lines and fill), `contour_l` (lines only), `contour_f` (fill only)" + \
                 "\n\t3D: `scatter3d`" + \
                 "\n\nExamples\n--------" + \
                 "\nEvolution plot of your progress coordinate:" + \
                 "\n\t$ wedap -W west.h5 (default) -dt evolution (default) -X pcoord (default)" + \
                 "\n\n1D average probability distribution of your progress coordinate:" + \
                 "\n\t$ wedap -W west.h5 -dt average -pm line" + \
                 "\n\n2D average probability distribution of pcoord 1 and 2:" + \
                 "\n\t$ wedap -W west.h5 -dt average -X pcoord (default) -Xi 0 (default) -Y pcoord -Yi 1" + \
                 "\n\n1D instant probability distribution of an aux dataset:" + \
                 "\n\t$ wedap -W west.h5 -dt instant -pm line -X auxname" + \
                 "\n\n3D scatter of your progress coordinates and aux data" + \
                 "\n\t$ wedap -W west.h5 -dt average -pm scatter3d -X pcoord -Xi 0 -Y pcoord -Yi 1 -Z auxname -Zi 0" + \
                 "\n\n2D average contour plot of 2 aux datasets for iterations 100 to 200 with probability limits in kcal/mol." + \
                 "\n\t$ wedap -dt average -pm contour -X auxname -Y auxname -fi 100 -li 200 --pmin 0 --pmax 20 --p-units kcal"

    # create argument parser (gooey based if available)
    if gooey is None:
        parser = argparse.ArgumentParser(description=wedap_desc, 
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    else:
        parser = gooey.GooeyParser(description=wedap_desc, 
                                   formatter_class=argparse.RawDescriptionHelpFormatter)

    ##########################################################
    ############### REQUIRED ARGUMENTS #######################
    ##########################################################

    # create new group for required args 
#     required_args = parser.add_argument_group("Required Arguments") 

#     # create file flag  
#     required_args.add_argument("-h5", "--h5file", required = True, help = "The \
#         WESTPA west.h5 output file that will be analyzed.", action = "store", 
#         dest = "h5", type=str) 

    # specify the main group of args needed and shown on initial page
    # note that these are positional arguments to parser
    # so like $ plothist average
    # good for different tools, maybe good for movie
    #main = parser.add_subparsers(help="Main Arguments", dest="Main")
    # sub_main = main.add_parser('option1')
    # sub_main.add_argument("test")
    # sub_main2 = main.add_parser('option2')
    # sub_main2.add_argument("test")

    # test out gooey specific widgets
    required = parser.add_argument_group("Required Arguments")
    required.add_argument("-W", "-w", "--west", "--west-data", "-h5", "--h5file", 
        default="west.h5", action="store", dest="h5", type=str, nargs="*",
        help="The WESTPA west.h5 output file that will be analyzed. "
             "Default 'west.h5'.", 
        widget="FileChooser")

    ###########################################################
    ############### OPTIONAL ARGUMENTS ########################
    ###########################################################
    # nargs = '?' "One argument will be consumed from the command line if possible, 
        # and produced as a single item. If no command-line argument is present, 
        # the value from default will be produced."

    main = parser.add_argument_group("Main Arguments")
    optional = parser.add_argument_group("Optional Extra Arguments")

    main.add_argument("-dt", "--data-type", "--datatype", default="evolution", nargs="?",
                        dest="data_type", choices=("evolution", "average", "instant"),
                        help="Type of pdist dataset to generate, options are "
                             "'evolution' (1 dataset); " 
                             "'average' or 'instance' (1 or 2 or 3 datasets).",
                        type=str) 
    main.add_argument("-pm", "--plot-mode", "--plotmode", default="hist", nargs="?",
                        dest="plot_mode", choices=("bar", "line", "hist", "hist_l", "contour", 
                                                   "contour_l", "contour_f", "scatter3d", "hexbin3d"),
                        help="The type of plot desired.  "
                             "For 1D: bar or line. "
                             "For 2D: hist or contour, hist_l is a hist with contour lines, "
                             "contour_l is just contour lines, contour_f is just the fill, "
                             "contour is both lines and fill, note that any contour lines "
                             "plotted will default to cmap colors, so to have more distinctive "
                             "contour lines, include a --color arg, e.g. `--color k`. \n"
                             "For 3D, scatter3d or hexbin3d.",
                        type=str)
    main.add_argument("-X", "-x", "--Xname", "--xname", default="pcoord", nargs="?",
                        dest="Xname", 
                        help="Target data name for x axis. Default 'pcoord', "
                        "can also be any aux dataset name in your h5 file.",
                        type=str)
    main.add_argument("-Y", "-y", "--Yname", "--yname", default=None, nargs="?",
                        dest="Yname", 
                        help="Target data name for y axis. Default 'None', "
                        "can be 'pcoord' or any aux dataset name in your h5 file.",
                        type=str)
    main.add_argument("-Z", "-z", "--Zname", "--zname", default=None, nargs="?",
                        dest="Zname", 
                        help="Target data name for z axis. Must use 'scatter3d' "
                        "for 'plot_mode'. Can be 'pcoord' or any aux dataset name "
                        "in your h5 file.",
                        type=str)
    main.add_argument("-Xi", "-xi", "--Xindex", "--xindex", default=0, nargs="?", type=int,
                        dest="Xindex", help="Index in third dimension for >2D datasets.")
    main.add_argument("-Yi", "-yi", "--Yindex", "--yindex", default=0, nargs="?", type=int,
                        dest="Yindex", help="Index in third dimension for >2D datasets.")
    main.add_argument("-Zi", "-zi", "--Zindex", "--zindex", default=0, nargs="?", type=int,
                        dest="Zindex", help="Index in third dimension for >2D datasets.")
    main.add_argument("-o", "--output", default=None,
                        dest="output_path",
                        help="The filename to which the plot will be saved. "
                             "Various image formats are available. You " 
                             "may choose one by specifying an extension. "
                             "\nLeave this empty if you don't want to save "
                             "the plot to a serperate file.",
                        type=str)
    # begin optional arg group
    optional.add_argument("-fi", "--first-iter", default=1, nargs="?",
                        dest="first_iter",
                        help="Plot data starting at iteration FIRST_ITER. "
                             "By default, plot data starting at the first "
                             "iteration in the specified west.h5 file.",
                        type=int)
    optional.add_argument("-li", "--last-iter", default=None, nargs="?",
                        dest="last_iter",
                        help="Plot data up to and including iteration LAST_ITER. "
                             "By default, plot data up to and including the last "
                             "iteration in the specified H5 file.",
                        type=int)
    optional.add_argument("-si", "--step-iter", default=1, nargs="?",
                        dest="step_iter",
                        help="Only use every step_iter size iteration intervals of the data "
                             "e.g. --step-iter 10 for every 10 iterations. Default 1.",
                        type=int)
    # *: a flexible number of values, which will be gathered into a list
    # +: like *, but requiring at least one value
    optional.add_argument("--bins", default=[100, 100], nargs="+",
                        dest="bins",
                        help="Use BINS number of bins for histogramming X and Y. "
                             "Divide the range between the minimum and maximum "
                             "observed values into this many bins. Must input 1 bin "
                             "value per dimension (e.g. X and Y so '100 100')",
                        type=int)
    optional.add_argument("-hrx", "--histrange-x", default=None, nargs=2,
                          dest="histrange_x",
                          help="Ranges to consider for the x-axis, input "
                               "2 space-seperated floats : LB UB",
                          type=float)
    optional.add_argument("-hry", "--histrange-y", default=None, nargs=2,
                          dest="histrange_y",
                          help="Ranges to consider for the y-axis, input "
                               "2 space-seperated floats : LB UB",
                          type=float)
    optional.add_argument("--pmin", "--p-min", default=None, nargs="?",
                        dest="p_min",
                        help="The minimum probability value limit. "
                             "This determines the cbar limits and contour levels.",
                        type=float)
    optional.add_argument("--pmax", "--p-max",default=None, nargs="?",
                        dest="p_max",
                        help="The maximum probability limit value. "
                             "This determines the cbar limits and contour levels.",
                        type=float)
    optional.add_argument("-pu", "--p-units", "--punits", default="kT", nargs="?",
                        dest="p_units", choices=("kT", "kcal", "raw", "raw_norm"),
                        help="Can be 'kT' (default), 'kcal', 'raw', or 'raw_norm'"
                             "kT = -lnP, kcal/mol = -RT(lnP), where RT=0.5922 at T(298K). "
                             "'raw' is the raw probabilities and "
                             "'raw_norm' is the raw probabilities P(max) normalized.",
                        type=str)
    optional.add_argument("-T", "--temp", default=298, nargs="?",
                        dest="T", help="Used with kcal/mol 'p-units'.",
                        type=int)
    optional.add_argument("-ci", "--contour-interval", default=1, nargs="?",
                        dest="contour_interval",
                        help="If using plot-mode contour, "
                             "This sets the interval of contour level.",
                        type=float)
    optional.add_argument("-cl", "--contour-levels", default=None, nargs="*",
                        dest="contour_levels",
                        help="If using plot-mode contour, "
                             "This overrides and sets the contour levels manually.",
                        type=float)
    optional.add_argument("-sl", "--smoothing-level", default=None, 
                        dest="smoothing_level",
                        help="Smooth data (plotted as histogram or contour"
                             " levels) using a gaussian filter with sigma="
                             "SMOOTHING_LEVEL.",
                        type=float)
    optional.add_argument("-sci", "--scatter-iterval", default=10, nargs="?",
                        dest="scatter_interval",
                        help="Adjust to use less data for scatter plots.",
                        type=int)
    optional.add_argument("-scs", "--scatter-s", default=1, nargs="?",
                        dest="scatter_s",
                        help="Adjust scatter plot marker size",
                        type=float)
    optional.add_argument("-hbg", "--hexbin-grid", default=100, nargs="?",
                        dest="hexbin_grid",
                        help="Adjusts hexbin gridsize parameters.",
                        type=int)
    # TODO: is there a better way to do this? 
    #optional.add_argument("--weighted", default=True, action="store_true",
    #                      help="Use weights from WE.")
    optional.add_argument("-nw", "--not-weighted",
                          help="Include this to not use WE weights.",
                          dest="not_weighted", action="store_true")
    # optional.add_argument("--weighted", default=True, 
    #                       action=argparse.BooleanOptionalAction)
    # * args is flexible number of values, which will be gathered into a list
    optional.add_argument("-sb", "--skip-basis", default=None, nargs="*",
                          dest="skip_basis",
                          help="List of binary values for skipping basis states, "
                               "e.g. 0 1 1 to skip all bstates except for first.",
                          type=int)
    optional.add_argument("-jp", "--jointplot", "--joint-plot", default=False,
                          dest="jointplot",
                          help="Optionally include marginal plots to create "
                               "a joint plot from 2D pdist.",
                          action="store_true")
    optional.add_argument("-3d", "--proj3d", default=False,
                          dest="proj3d",
                          help="Make a 3d projection plot, works with contour or scatter plots.",
                          action="store_true")
    optional.add_argument("-4d", "--proj4d", default=False,
                          dest="proj4d",
                          help="Make a 4d projection plot, must have Cname. " +
                               "only works with scatter plots.",
                          action="store_true")
    optional.add_argument("-C", "-c", "--Cname", "--cname", default=None, nargs="?",
                         dest="Cname", 
                         help="Target data name for cbar of proj3d. Must use 'scatter3d' "
                         "for 'plot_mode'. Can be 'pcoord' or any aux dataset name "
                         "in your h5 file.",
                         type=str)
    optional.add_argument("-Ci", "--Cindex", "--cindex", default=0, nargs="?", type=int,
                           dest="Cindex", help="Index in third dimension for >2D datasets.")

    # create optional flag to output everything to console screen
    # optional.add_argument("-ots", "--output_to_screen", default=True,
    #                     dest = "output_to_screen",
    #                     help = "Outputs plot to screen. True (default) or False", 
    #                     action= "store_true") 
    optional.add_argument("-nots", "--no-output-to-screen",
                        dest = "no_output_to_screen",
                        help = "Include this argument to not output the plot to "
                        "your display.", 
                        action= "store_true") 
    optional.add_argument("-npb", "--no-progress-bar",
                        dest = "no_pbar",
                        help = "Include this argument to not output the tqdm progress bar.",
                        action= "store_true")

    # plot tracing arg group
    trace = parser.add_argument_group("Optional Plot Tracing", 
                                       description="Plot a trace on top of the pdist.")
    trace_group = trace.add_mutually_exclusive_group()
    # type to float for val inside tuple, 
    # and nargs to 2 since it is interpreted as a 2 item tuple or list
    trace_group.add_argument("--trace-seg", default=None, nargs=2,
                             dest="trace_seg",
                             help="Trace and plot a single continuous trajectory based "
                                 "off of 2 space-seperated ints : iteration segment",
                             type=int)
    trace_group.add_argument("--trace-val", default=None, nargs=2,
                             dest="trace_val",
                             help="Trace and plot a single continuous trajectory based "
                                  "off of 2 space-seprated floats : Xvalue Yvalue",
                             type=float)

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
                             "Edit the wedap/styles/default.mplstyle file to "
                             "change default wedap plotting style options.",
                        type=str)
    # TODO: prob cant use custom outside of list
    formatting.add_argument("--cmap", default="viridis", nargs="?",
                        dest="cmap", help="mpl colormap name.", type=str)
    formatting.add_argument("--color",
                        dest="color", help="Color for 1D plots, contour lines, and trace plots.",
                        widget="ColourChooser")
    formatting.add_argument("--linewidth", "-lw", default=None, nargs="?",
                        dest="linewidth", help="Linewidth for 1D plots, contour lines, and trace plots.",
                        type=float)
    formatting.add_argument("--linestyle", "-ls", default="-", nargs="?",
                        dest="linestyle", help="Linestyle for 1D plots, contour lines, and trace plots.",
                        type=str)
    formatting.add_argument("--xlabel", dest="xlabel", type=str)
    formatting.add_argument("--xlim", help="LB UB", dest="xlim", nargs=2, type=float)
    formatting.add_argument("--ylabel", dest="ylabel", type=str)
    formatting.add_argument("--ylim", help="LB UB", dest="ylim", nargs=2, type=float)
    formatting.add_argument("--title", dest="title", type=str)
    formatting.add_argument("--suptitle", dest="suptitle", type=str)
    formatting.add_argument("--cbar-label", dest="cbar_label", type=str)
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