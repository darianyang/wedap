"""
Main call.
"""

from command_line import *
from h5_plot_main import *

# if python file is being used 
if __name__ == '__main__': 
    
    # TODO: chicken and egg problem, when to initialize the parser?
    f = h5py.File("data/west.h5", mode="r")
    aux = list(f[f"iterations/iter_00000001/auxdata/"])

    """
    Command line
    """
    # Create command line arguments with argparse
    argument_parser = create_cmd_arguments(aux)
    # Retrieve list of args
    args_list = handle_command_line(argument_parser)

    """
    Generate pdist and plot it
    """
    X, Y, Z = pdist_to_normhist(args_list)
    plot_normhist(X, Y, args_list, norm_hist=Z) #TODO: adjust arg/kwarg order

    """
    Trace (Optional Argument)
    """
    #if args_list.trace is True:

    """
    Show and/or save the final plot
    """
    if args_list.output_path is True:
        plt.savefig(args_list.output_path, dpi=300, transparent=True)
    if args_list.output_to_screen is True:
        plt.show()
