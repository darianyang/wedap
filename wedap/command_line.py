"""
Functions for handling command-line input using argparse module.
"""

import argparse
import os
import sys

#from gooey import Gooey

#@Gooey(optional_cols=4, default_size=(900, 700))
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

    # create argument parser 
    parser = argparse.ArgumentParser(description = 
        "Weighted Ensemble data analysis and plotting (WE-dap). \n"
        "Given an input west.h5 file from a successful WESTPA simulation, prepare "
        "probability distributions and plots.")

    # TODO: put requried first

    ###########################################################
    ############### OPTIONAL ARGUMENTS ########################
    ###########################################################
    # nargs = '?' "One argument will be consumed from the command line if possible, 
        # and produced as a single item. If no command-line argument is present, 
        # the value from default will be produced."

    parser.add_argument("--first-iter", default=1, nargs="?",
                        dest="first_iter",
                        help="Plot data starting at iteration FIRST_ITER."
                             "By default, plot data starting at the first"
                             "iteration in the specified west.h5 file.",
                        type=int)
    parser.add_argument("--last-iter", default=None, nargs="?",
                        dest="last_iter",
                        help="Plot data up to and including iteration LAST_ITER."
                             "By default, plot data up to and including the last "
                             "iteration in the specified w_pdist file.",
                        type=int)
    parser.add_argument("--bins", default=100, nargs="?",
                        dest="bins",
                        help="Use BINS number of bins for histogramming "
                             "Divide the range between the minimum and maximum "
                             "observed values into this many bins",
                        type=int)
    parser.add_argument("--p_max", default=None, nargs="?",
                        dest="p_max",
                        help="The maximum probability limit value."
                             "This determines the cbar limits and contours levels.",
                        type=int)
    parser.add_argument("--p_units", default="kT", nargs="?",
                        dest="p_units", choices=("kT", "kcal"),
                        help="Can be 'kT' (default) or 'kcal'." # TODO: temp arg
                             "kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at 298K.",
                        type=str)
    parser.add_argument("--data_type", default="evolution", nargs="?",
                        dest="data_type", choices=("evolution", "average", "instance"),
                        help="Type of pdist dataset to generate, options are"
                             "'evolution' (1 dataset);" 
                             "'average' or 'instance' (1 or 2 datasets)",
                        type=str) 
    parser.add_argument("--plot_type", default="heat", nargs="?",
                        dest="plot_type", choices=("heat", "contour"),
                        help="The type of plot desired, current options are"
                             "'heat' and 'contour'.",
                        type=str)
    parser.add_argument("--cmap", default="viridis", nargs="?",
                        dest="cmap", choices=("viridis", "afmhot", "gnuplot_r"),
                        help="mpl colormap style.",
                        type=str)
    # TODO: could make choices tuple with the available aux values from the h5 file
    parser.add_argument("--aux_x", default=None, nargs="?", #TODO: default to pcoord w/ none
                        dest="aux_x", #choices=aux, TODO
                        help="Target data for x axis.",
                        type=str)
    parser.add_argument("--aux_y", default=None, nargs="?", #TODO: default to pcoord w/ none
                        dest="aux_y", #choices=aux, TODO
                        help="Target data for x axis.",
                        type=str)
    parser.add_argument("--output", default=None,
                        dest="output_path",
                        help="The filename to which the plot will be saved."
                             "Various image formats are available.  You " 
                             "may choose one by specifying an extension",
                        type=str)
    trace_group = parser.add_mutually_exclusive_group()
    # type to float for val inside tuple, and nargs to 2 since it is interpreted as a 2 item tuple or list
    trace_group.add_argument("--trace_seg", default=None, nargs=2,
                             dest="trace_seg",
                             help="Trace and plot a single continuous trajectory based"
                                 "off of 2 int values : iteration and segment",
                             type=int)
    trace_group.add_argument("--trace_val", default=None, nargs=2,
                             dest="trace_val",
                             help="Trace and plot a single continuous trajectory based"
                                  "off of 2 float : aux_x value and aux_y value",
                             type=float)

    # TODO
    # parser.add_argument('--smooth-data', default = None, 
    #                     dest='data_smoothing_level',
    #                     help='Smooth data (plotted as histogram or contour'
    #                             ' levels) using a gaussian filter with sigma='
    #                             'DATA_SMOOTHING_LEVEL.',
    #                     type=float)

    # create optional flag to output everything to console screen 
    parser.add_argument("--outputToScreen", default=True,
                        dest = "output_to_screen",
                        help = "Outputs plot to screen", 
                        action= "store_true") 

    ##########################################################
    ############### REQUIRED ARGUMENTS #######################
    ##########################################################

    # create new group for required args 
    required_args = parser.add_argument_group("Required Arguments") 

    # create file flag  
    required_args.add_argument("-h5", "--h5file", required = True, help = "The \
        WESTPA west.h5 output file that will be analyzed.", action = "store", 
        dest = "h5", type=str) 

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
        Contains all arguments passed into EnsembleOptimizer.
    
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