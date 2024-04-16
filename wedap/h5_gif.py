"""
Helper function for making gifs.
"""
from .h5_plot import *
import gif
from tqdm.auto import tqdm
import matplotlib as mpl

def make_gif(first_iter, last_iter, step_iter=1, avg_plus=100, 
             duration=50, gif_out="example.gif", **kwargs):
    """
    Convenience function for gif making.
    Note that this is tailored for making average pdist plots.

    Parameters
    ----------
    we : wedap H5_Plot object
        Input this object and the H5_Plot.plot() method is ran to generate the plot
        for the gif at the specified iters
    first_iter : int
        Where to start the gif.
    last_iter : int
        Where to end the gif. Important here is that you
        make sure avg_plus + last_iter does not exceed the total amount of iters
        you have available in the h5 file.
    step_iter : int
        Interval for looping the first to last iter requested.
    avg_plus : int
        The +range of interations for each iter in range(first,last,step).
        So as the loop progresses, avg_plus is added to each iter to make the 
        range that the average pdist is taken from. Important here is that you
        make sure avg_plus + last_iter does not exceed the total amount of iters
        you have available in the h5 file. If you set avg_plus to 0, it will
        make instant plots of each iter in the range requested.
    duration : int
        Duration in milliseconds between frames of the gif, default 50ms.
    gif_out : str
        Out path to created gif file, default 'example.gif'.
    **kwargs
        Can be useful to input dictionary of kwargs for H5_Plot init.
        E.g. can put xlim, xlabel, grid, etc.
    """
    # plots for a gif should not be saved with any transparency
    mpl.rcParams["savefig.transparent"] = False
    mpl.rcParams["savefig.facecolor"] = "white"

    # decorate a plot function with @gif.frame (return not required):
    @gif.frame
    def plot_for_gif(iteration):
        """
        Make a single frame for a gif of multiple wedap plots.

        Parameters
        ----------
        iteration : int
            Plot a specific iteration.
        """
        plot_defaults = {"first_iter" : iteration,
                        "last_iter" : iteration + avg_plus,
                        "title" : f"WE Iteration {iteration} to {iteration + avg_plus}",
                        "no_pbar" : True,
                        }

        # combine default gif settings with user input
        # since the defaults are second, it should replace the same keys in input options
        plot_options = {**kwargs, **plot_defaults}

        # call plotting method to generate the plot and a frame of the gif
        H5_Plot(**plot_options).plot()

    # build a bunch of "frames"
    # having at least 100 frames makes for a good length gif
    frames = []
    # set the range to be the iterations at a specified interval
    for iter in tqdm(range(first_iter, last_iter, step_iter)):
        frame = plot_for_gif(iter)
        frames.append(frame)

    # specify the duration between frames (milliseconds) and save to file:
    gif.save(frames, gif_out, duration=duration)