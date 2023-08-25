"""
Plot MD pdists.

TODO: specify option for:
    timeseries (option for KDE side plot and option for stdev vs all reps)
    pdist (1D hist, 1D KDE, + others from H5_Plot)
    others?
"""

from .md_pdist import MD_Pdist

# relative import of wedap H5_Plot class (TODO: alt import?)
from wedap import H5_Plot

# subclass H5_Plot to have access to those pdist plots
# also multiple inheritance for features from MD_Pdist as well
class MD_Plot(H5_Plot, MD_Pdist):
    """
    These methods provide various plotting options for pdist data.
    """
    def __init__(self, *args, **kwargs):

        # for jointplots, save original p_units and run pdist with raw
        # only if jointplot var exists and is True
        if "jointplot" in kwargs:
            # separate since if it doesn't exist can't index to check if True
            if kwargs["jointplot"] is True:
                # with user specified p_units, save it
                if "p_units" in kwargs:
                    og_p_units = kwargs["p_units"]
                # always set p_units to 'raw' for joint plots
                kwargs["p_units"] = "raw"

        # initialize md pdist
        MD_Pdist.__init__(self, *args, **kwargs)
        # generate xyz arrays
        self.X, self.Y, self.Z = self.pdist()

        # change back to original p_units
        # only if jointplot var exists and is True
        if "jointplot" in kwargs:
            # separate since if it doesn't exist can't index to check if True
            if kwargs["jointplot"] is True:
                # if the original p_units from user were provided and saved
                if "p_units" in kwargs and "og_p_units" in locals():
                    kwargs["p_units"] = og_p_units
                    self.p_units = og_p_units
                # otherwise default to kT
                else:
                    kwargs["p_units"] = "kT"
                    self.p_units = "kT"

        # run H5_Plot initialization
        H5_Plot.__init__(self, self.X, self.Y, self.Z, *args, **kwargs)