"""
Main call.
"""

from command_line import *
from h5_plot_main import *

# if python file is being used 
if __name__ == '__main__': 
    
    """
    Command line
    """
    # Create command line arguments with argparse
    argument_parser = create_cmd_arguments()
    # Retrieve list of args
    args_list = handle_command_line(argument_parser)

    """
    Generate pdist and plot
    """
    X, Y, Z = pdist_to_normhist(args_list)
    plot_normhist(X, Y, Z, args_list)

    """
    Trace (Optional Argument)
    """
    if args_list.trace is True:
