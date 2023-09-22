"""
Convert MD analyzed data to pdists.

TODO:
    Maybe could include an arg for custom weights
"""

import sys
import numpy as np
from wedap import H5_Pdist

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

class MD_Pdist(H5_Pdist):
    """
    These class methods generate probability distributions from input MD data files.
    """
    # TODO: is setting aux_y to None the best approach to 1D plot settings?
    # TODO: add step-iter
    def __init__(self, data_type=None, Xname=None, Xindex=1, Yname=None, Yindex=1, 
                 Zname=None, Zindex=1, Xinterval=1, Yinterval=1, Zinterval=1, data_proc=None, 
                 first_iter=1, last_iter=None, bins=(100,100), p_units='kT', T=298, 
                 histrange_x=None, histrange_y=None, no_pbar=False, timescale=10**6,
                 *args, **kwargs):
        """
        TODO: add XYZ interval to proc (default 1)

        Parameters
        ----------
        data_type : str
            'time' for 1 dataset timeseries, or 'pdist' for everything else.
        Xname : str or list of str
            target data for x axis, default None.
        Xindex : int
            If X.ndim > 2, use this to index.
        Yname : str or list of str
            target data for y axis, default None.
        Yindex : int
            If Y.ndim > 2, use this to index.
        Zname : str or list of str
            target data for z axis, default None. 
            Use this if you want to use a dataset instead of pdist for Z axis.
            This will be best plotted as a scatter plot with Z as the marker color.
            Instead of returning the pdist, only the XYZ datasets will be returned.
            This is becasue the weights/pdist isn't considered.
        Zindex : int
            If Z.ndim > 2, use this to index.
        Xinterval, Yinterval, Zinterval : int
            Interval for processing dataset. E.g. 10 = every 10 frames.
        data_proc : function or tuple of functions
            Of the form f(data) where data has rows=segments, columns=frames until tau, depth=data dims.
            The input function must return a processed array of the same shape and formatting.
        first_iter : int
            Default start plot at iteration 1 data.
        last_iter : int
            Last iteration data to include, default is the last recorded iteration in the west.h5 file. 
            Note that `instant` type pdists only depend on last_iter.
        bins : tuple of ints (TODO: maybe the tuple isn't user friendly for 1 dim?)
            Histogram bins in pdist data to be generated for x and y datasets, default both 100.
        p_units : str
            Can be 'kT' (default), 'kcal', 'raw', or 'raw_norm'.
            kT = -lnP, kcal/mol = -RT(lnP), where RT = 0.5922 at `T` Kelvin.
            'raw' is the raw probabilities and 'raw_norm' is the raw probabilities P(max) normalized.
        T : int
            Temperature if using kcal/mol.
        histrange_x, histrange_y : list or tuple of 2 floats or ints
            Optionally put custom bin ranges.
        no_pbar : bool
            Optionally do not include the progress bar for pdist generation.
        timescale : int
            Default ps to µs (10**6). Converts frames to time.
        TODO: maybe also binsfromexpression?
        """
        if data_type is None or data_type not in ["time", "pdist"]:
            raise ValueError("Must input valid data_type str: `time` or `pdist`")
        else:
            self.data_type = data_type
        self.p_units = p_units
        self.T = T

        self.Xname = Xname
        self.Xindex = Xindex
        self.Yname = Yname
        self.Yindex = Yindex
        self.Zname = Zname
        self.Zindex = Zindex
        
        self.Xinterval = Xinterval
        self.Yinterval = Yinterval
        self.Zinterval = Zinterval

        # raw data processing function
        # TODO: allow for 1-3 functions as tuple input, right now one function only
        self.data_proc = data_proc

        # default to last
        if last_iter is not None:
            self.last_iter = last_iter
        elif last_iter is None:
            pass # entire trajectory
        self.first_iter = first_iter

        self.bins = bins
        self.histrange_x = histrange_x
        self.histrange_y = histrange_y
        self.no_pbar = no_pbar
        self.timescale = timescale

    def _get_md_data(self, names, index, interval):
        """
        Return MD data in array.
        """
        # grab dataset (make sure 2D for proper indexing)
        # TODO: switch to pre-cast array?
        data = []

        # if the input file isn't already a list, make it a 1 item list
        if not isinstance(names, list):
            names = [names]

        # handle multiple file name list
        # TODO: handle direct ndarray inputs as well as file str
        for name in names:
            # for .npy binary files or pkl files
            if name[-4:] in [".npy", ".npz", ".pkl"]:
                data_item = np.load(name, allow_pickle=True)
            else:
                # TODO: note that genfromtxt can handle multiple items in list, could this help?
                data_item = np.genfromtxt(name)
            # for 1D datasets, need to standardize to 2D, but as a new column
            if data_item.ndim < 2:
                data_item = data_item[:, np.newaxis]
            data.append(data_item[::interval, index])
        # combine into a single array
        data = np.concatenate(data)

        return data

    def timeseries(self):
        """
        Returns
        -------
        X : ndarray
        Y : ndarray
        """
        # could get time from frame column
        #time = np.concatenate([np.genfromtxt(i)[::self.Xinterval, 0] for i in self.Xname])

        X = self._get_md_data(self.Xname, self.Xindex, self.Xinterval)

        # or can just get it from n rows
        time = np.arange(0, X.shape[0], self.Xinterval)
        time = np.divide(time, self.timescale)

        return time, X

    def pdist_1d(self):
        """
        Returns
        -------
        X : ndarray
        Y : ndarray
        """
        #X = np.concatenate([np.genfromtxt(i)[::self.Xinterval, self.Xindex] for i in self.Xname])
        X = self._get_md_data(self.Xname, self.Xindex, self.Xinterval)
    
        # get rid of nan values: return array without (not) True nan values
        #X = X[np.logical_not(np.isnan(X))]

        # numpy equivalent to: ax.hist2d(c2[:,1], aux)
        hist, x_edges = np.histogram(X, bins=self.bins[0], range=self.histrange_x)
        # let each row list bins with common y range
        hist = np.transpose(hist)
        # convert histogram counts to p_units
        hist = self._normalize(hist, self.p_units)
        # get bin midpoints
        midpoints_x = (x_edges[:-1] + x_edges[1:]) / 2

        return midpoints_x, hist

    def pdist_2d(self):
        """
        Returns
        -------
        X : ndarray
        Y : ndarray
        Z : ndarray
        """
        X = self._get_md_data(self.Xname, self.Xindex, self.Xinterval)
        Y = self._get_md_data(self.Yname, self.Yindex, self.Yinterval)
    
        try:
            # numpy equivalent to: ax.hist2d(c2[:,1], aux)
            hist, x_edges, y_edges = np.histogram2d(X, Y, bins=self.bins, 
                                                    range=[self.histrange_x, self.histrange_y])
        except ValueError as e:
            message = f"{e}: do the size of the x {X.shape} and y {Y.shape} inputs match? "
            message += f"--Xinterval and --Yinterval can be adjusted."
            raise ValueError(message)
        
        # let each row list bins with common y range
        hist = np.transpose(hist)
        # convert histogram counts to p_units
        hist = self._normalize(hist, self.p_units)
        # get bin midpoints
        midpoints_x = (x_edges[:-1] + x_edges[1:]) / 2
        midpoints_y = (y_edges[:-1] + y_edges[1:]) / 2

        return midpoints_x, midpoints_y, hist
        
    def pdist_3d(self):
        """
        Returns
        -------
        X : ndarray
        Y : ndarray
        Z : ndarray
        """
        X = self._get_md_data(self.Xname, self.Xindex, self.Xinterval)
        Y = self._get_md_data(self.Yname, self.Yindex, self.Yinterval)
        Z = self._get_md_data(self.Zname, self.Zindex, self.Zinterval)

        return X, Y, Z

    def pdist(self):
        """
        Main public method with pdist generation controls.
        """ 
        # return timeseries data
        if self.data_type == "time":
            X, Y = self.timeseries()
            return X, Y, np.ones((X.shape[0]))
        # return pdist data
        # TODO: t/e block to catch mismatched data lengths and print lengths?
        elif self.data_type == "pdist":
            # for 3D datasets, e.g. scatter
            if self.Yname and self.Zname:
                return self.pdist_3d()
            # if 2D, return 2D dataset
            if self.Yname:
                return self.pdist_2d()
            # otherwise return 1D
            else:
                X, Y = self.pdist_1d()
                # TODO: fake 3D output for now to accomodate main.py
                return X, Y, np.ones(X.shape[0])
