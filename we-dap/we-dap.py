"""
Main call.

"""

from command_line import *
from h5_plot_main import *

from search_aux import *
from h5_plot_trace import *

# if python file is being used 
if __name__ == '__main__': 
    
    # TODO: chicken and egg problem, when to initialize the parser?
    #f = h5py.File("data/west.h5", mode="r")
    #aux = list(f[f"iterations/iter_00000001/auxdata/"])

    """
    Command line
    """
    # TODO: it may be more interpretable if I add each args_list.value as arguments to
        # the default functions and keep the original args, I think this would allow 
        # for a better python API down the line
    # Create command line arguments with argparse
    argument_parser = create_cmd_arguments()
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
    if args_list.trace_seg is not None:
        plot_trace(args_list.h5, args_list.trace_seg, args_list.aux_x, args_list.aux_y)
    if args_list.trace_val is not None:
        # for 1A43 V02: C2 and Dist M2-M1 - minima at val = 53° and 2.8A is alt minima = i173 s70
        # for demo: can use x = 53 and y = 2.7 or 2.6
        iter, seg = search_aux_xy_nn(args_list.h5, args_list.aux_x, args_list.aux_y, 
                                    # TODO: update to aux_x aux_y tuple
                                    args_list.trace_val[0], args_list.trace_val[1], args_list.last_iter)
        plot_trace(args_list.h5, (iter,seg), args_list.aux_x, args_list.aux_y)

    """
    Plot formatting
    """
    plt.xlabel(args_list.aux_x)
    if args_list.aux_y:
        plt.ylabel(args_list.aux_y)
    if args_list.data_type == "evolution":
        plt.ylabel("WE Iteration")

    """
    Show and/or save the final plot
    """
    plt.tight_layout()
    # TODO: the save fig option produces a choppy image
    if args_list.output_path is not None:
        plt.savefig(args_list.output_path, dpi=300, transparent=True)
    if args_list.output_to_screen is True:
        plt.show()
