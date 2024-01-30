"""
Functions for handling command-line input using argparse module.
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
        program_name='mdap',
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
    arguments for `mdap`. 

    Parameters 
    ----------

    Returns
    -------
    argparse.ArgumentParser: 
        An ArgumentParser that is used to retrieve command line arguments. 
    """
    mdap_desc = "======================================================= \n" + \
                "=== molecular dynamics analysis and plotting (mdap) === \n" + \
                "======================================================= \n" + \
                "\nGiven an input (pre-calcualated) dataset from standard MD simulations, " + \
                "prepare probability distributions and plots. " + \
                "Input data must be in >=2 column format: \n" + \
                "# use hashs at top of data file to indicate comments (skipped) \n" + \
                "COL1:Frame | COL2:Data | COL3:Data... \n\n" + \
                "\nSee the documentation for usage and examples: https://darianyang.github.io/wedap " + \
                "\n\n" + \
                "mdap can be used with 2 different --data-type (-dt) args: " + \
                "\n\t`pdist` (default) and `time`" + \
                "\n\nAvailable --plot-mode (-pm) options are: " + \
                "\n\t1D: `line`" + \
                "\n\t2D: `hist` (default), `hist_l` (hist with contour lines), " + \
                "\n\t    `contour` (lines and fill), `contour_l` (lines only), `contour_f` (fill only)" + \
                "\n\t3D: `scatter3d`" + \
                "\n\nExamples\n--------" + \
                "\nTime evolution plot:" + \
                "\n\t$ mdap -X input_data.dat -dt time -pm line" + \
                "\n\n1D average probability distribution:" + \
                "\n\t$ mdap -X input_data.dat -dt pdist -pm line" + \
                "\n\n2D average probability distribution of datasets 1 and 2 and default column index:" + \
                "\n\t$ mdap -X input_data_0.dat -Y input_data_1.dat -Xi 1 -Yi 1" + \
                "\n\n3D scatter of three input datasets:" + \
                "\n\t$ mdap -pm scatter3d -X input_data_0.dat -Y input_data_1.dat -Z input_data_2.dat" + \
                "\n\n2D average contour plot of 2 input datasets probability limits in kcal/mol:" + \
                "\n\t$ mdap -pm contour -X input_data_0.dat -Y input_data_1.dat --pmin 0 --pmax 5 --p-units kcal"

    # create argument parser (gooey based if available)
    if gooey is None:
        parser = argparse.ArgumentParser(description=mdap_desc, 
                                        formatter_class=argparse.RawDescriptionHelpFormatter)
    else:
        parser = gooey.GooeyParser(description=mdap_desc, 
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
#     required = parser.add_argument_group("Required Arguments")
#     required.add_argument("-h5", "--h5file", #required=True, nargs="?",
#         default="west.h5", action="store", dest="h5", type=str,
#         help="The WESTPA west.h5 output file that will be analyzed. "
#              "Default 'west.h5'.", 
#         widget="FileChooser")

    ###########################################################
    ############### OPTIONAL ARGUMENTS ########################
    ###########################################################
    # nargs = '?' "One argument will be consumed from the command line if possible, 
        # and produced as a single item. If no command-line argument is present, 
        # the value from default will be produced."

    main = parser.add_argument_group("Main Arguments")
    optional = parser.add_argument_group("Optional Extra Arguments")

    main.add_argument("-dt", "--data-type", "--datatype", default="pdist", nargs="?",
                        dest="data_type", choices=("time", "pdist"),
                        help="Type of pdist dataset to generate, options are ",
                             # TODO
                        type=str) 
    main.add_argument("-pm", "--plot-mode", "--plotmode", default="hist", nargs="?",
                        dest="plot_mode", choices=("hist", "hist_l", "contour", "contour_l", 
                                                   "contour_f", "bar", "line", "scatter3d", "hexbin3d"),
                        help="The type of plot desired.  "
                             "e.g. line for 1D, hist or contour for 2D and scatter3d for 3D.",
                        type=str)
    # TODO: allow a list of files, then multiple replicates can be addressed
    # * args is flexible number of values, which will be gathered into a list
    main.add_argument("-X", "-x", "--Xname", "--xname", default=None, nargs="*",
                        dest="Xname",
                        help="Target data name for x axis",
                        type=str)
    main.add_argument("-Y", "-y", "--Yname", "--yname", default=None, nargs="*",
                        dest="Yname",
                        help="Target data name for y axis.",
                        type=str)
    main.add_argument("-Z", "-z", "--Zname", "--zname", default=None, nargs="*",
                        dest="Zname", 
                        help="Target data name for z axis. Must use 'scatter3d' "
                        "for 'plot_mode'.",
                        type=str)
    # default to index 1 (2nd item of dataset)
    main.add_argument("-Xi", "-xi", "--Xindex", "--xindex", default=1, nargs="?", type=int,
                        dest="Xindex", help="Index in third dimension for >2D datasets.")
    main.add_argument("-Yi", "-yi", "--Yindex", "--yindex", default=1, nargs="?", type=int,
                        dest="Yindex", help="Index in third dimension for >2D datasets.")
    main.add_argument("-Zi", "-zi", "--Zindex", "--zindex", default=1, nargs="?", type=int,
                        dest="Zindex", help="Index in third dimension for >2D datasets.")
    # default to interval of 1 (process every frame)
    # TODO: update convention to be different from index
    main.add_argument("-Xint", "-xint", "--Xinterval", "--xinterval", default=1, nargs="?", type=int,
                        dest="Xinterval", help="Interval in third dimension for >2D datasets.")
    main.add_argument("-Yint", "-yint", "--Yinterval", "--yinterval", default=1, nargs="?", type=int,
                        dest="Yinterval", help="interval in third dimension for >2D datasets.")
    main.add_argument("-Zint", "-zint", "--Zinterval", "--zinterval", default=1, nargs="?", type=int,
                        dest="Zinterval", help="Interval in third dimension for >2D datasets.")
    main.add_argument("-hrx", "--histrange-x", default=None, nargs=2,
                      dest="histrange_x",
                      help="Ranges to consider for the x-axis, input "
                           "2 space-seperated floats : LB UB",
                      type=float)
    main.add_argument("-hry", "--histrange-y", default=None, nargs=2,
                      dest="histrange_y",
                      help="Ranges to consider for the y-axis, input "
                           "2 space-seperated floats : LB UB",
                      type=float)
    main.add_argument("-o", "--output", default=None,
                        dest="output_path",
                        help="The filename to which the plot will be saved. "
                             "Various image formats are available. You " 
                             "may choose one by specifying an extension. "
                             "\nLeave this empty if you don't want to save "
                             "the plot to a serperate file.",
                        type=str)
    # begin optional arg group
    # TODO: update to be first/last frame and add an interval arg
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
    # *: a flexible number of values, which will be gathered into a list
    # +: like *, but requiring at least one value
    optional.add_argument("--bins", default=[100, 100], nargs="+",
                        dest="bins",
                        help="Use BINS number of bins for histogramming X and Y. "
                             "Divide the range between the minimum and maximum "
                             "observed values into this many bins. Must input 1 bin "
                             "value per dimension (e.g. X and Y so '100 100')",
                        type=int)
    optional.add_argument("--pmin", default=None, nargs="?",
                        dest="p_min",
                        help="The minimum probability value limit. "
                             "This determines the cbar limits and contour levels.",
                        type=float)
    optional.add_argument("--pmax", default=None, nargs="?",
                        dest="p_max",
                        help="The maximum probability limit value. "
                             "This determines the cbar limits and contour levels.",
                        type=float)
    optional.add_argument("-pu", "--p-units", default="kT", nargs="?",
                        dest="p_units", choices=("kT", "kcal", "raw", "raw_norm"),
                        help="Can be 'kT' (default) or 'kcal'. "
                             "kT = -lnP, kcal/mol = -RT(lnP), "
                             "where RT=0.5922 at T(298K).",
                        type=str)
    optional.add_argument("-ts", "--timescale", "--time-scale", default=10e6, nargs="?",
                        dest="timescale",
                        help="Default ps to µs (10e6). Converts frames to time.",
                        type=float)
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
    optional.add_argument("-T", "--temp", default=298, nargs="?",
                        dest="T", help="Used with kcal/mol 'p-units'.",
                        type=int)
    optional.add_argument("-jp", "--joint-plot", default=False,
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

    # create optional flag to not output plot to console screen
    optional.add_argument("-nots", "--no-output-to-screen",
                        dest = "no_output_to_screen",
                        help = "Include this argument to not output the plot to "
                        "your display.", 
                        action= "store_true") 
    optional.add_argument("-npb", "--no-progress-bar",
                        dest = "no_pbar",
                        help = "Include this argument to not output the tqdm progress bar.",
                        action= "store_true")

    # plot tracing arg group (TODO: include this as trace-frame and can also trace-val)
    trace = parser.add_argument_group("Optional Plot Tracing", 
                                       description="Plot a trace on top of the pdist.")
    trace_group = trace.add_mutually_exclusive_group()
    # type to float for val inside tuple, 
    # and nargs to 2 since it is interpreted as a 2 item tuple or list
    trace_group.add_argument("--trace-frame", default=None, nargs=1,
                             dest="trace_frame",
                             help="Trace and plot a single continuous trajectory based "
                                 "off of 1 ints to specify frame number.",
                             type=int)
    trace_group.add_argument("--trace-val", default=None, nargs=2,
                             dest="trace_val",
                             help="Trace and plot the trajectory up this value based "
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