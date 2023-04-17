"""
Convert auxillary data recorded during WESTPA simulation and stored in west.h5 file
to various probability density datasets.

This script effectively replaces the need to use the native WESTPA plotting pipeline:
west.h5 --w_pdist(with --construct-dataset module.py)--> 
pdist.h5 --plothist(with --postprocess-functions hist_settings.py)--> plot.pdf

TODO: 
    - maybe add option to output pdist as file, this would speed up subsequent plotting
        of the same data.
    - add option for a list of equivalent h5 files, alternative to w_multi_west.
"""

# TEMP for trace plot (TODO)
import matplotlib.pyplot as plt
######

import h5py
import numpy as np
from tqdm.auto import tqdm

from scipy.spatial import KDTree

from warnings import warn

# for copying h5 file
import shutil

# Suppress divide-by-zero in log
np.seterr(divide='ignore', invalid='ignore')

# TODO: maybe can have the plot class take a pdist object as the input
#       then if I want to use a loaded pdist, easy to swap it
class H5_Pdist():
    """
    These class methods generate probability distributions from a WESTPA H5 file.
    """
    # TODO: is setting aux_y to None the best approach to 1D plot settings?
    # TODO: add step-iter
    def __init__(self, h5="west.h5", data_type=None, Xname="pcoord", Xindex=0, Yname=None, 
                 Yindex=0, Zname=None, Zindex=0, H5save_out=None, Xsave_name=None, Ysave_name=None,
                 Zsave_name=None, data_proc=None, first_iter=1, last_iter=None, bins=(100,100), 
                 p_units='kT', T=298, weighted=True, skip_basis=None, skip_basis_out=None,
                 histrange_x=None, histrange_y=None, no_pbar=False):
        """
        Parameters
        ----------
        h5 : str
            path to west.h5 file.
        data_type : str
            'evolution' (1 dataset); 'average' or 'instant' (1 or 2 datasets)
        Xname : str
            target data for x axis, default pcoord.
        Xindex : int
            If X.ndim > 2, use this to index.
        Yname : str
            target data for y axis, default None.
        Yindex : int
            If Y.ndim > 2, use this to index.
        Zname : str
            target data for z axis, default None. 
            Use this if you want to use a dataset instead of pdist for Z axis.
            This will be best plotted as a scatter plot with Z as the marker color.
            Instead of returning the pdist, only the XYZ datasets will be returned.
            This is becasue the weights/pdist isn't considered.
        Zindex : int
            If Z.ndim > 2, use this to index.
        H5save_out : str
            Paths to save a new H5 file with this dataset name.
            Right now it saves the requested X Y or Z data into a new aux_name.
            Note if you use this feature the input data must be the same shape and formatting as the other
            H5 file datasets. (TODO: organization?)
        Xsave_name, Ysave_name, Zsave_name : str
            Respective names to call the new dataset saved into the new H5 file.
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
        weighted : bool
            Default True, use WE segment weights in pdist calculation.
        skip_basis : list
            List of binaries for each basis state to determine if it is skipped.
            e.g. [0, 0, 1] would only consider the trajectory data from basis 
            states 1 and 2 but would skip basis state 3, applying zero weights.
        skip_basis_out : str
            Name of the outfile h5 file for optionally outputting new skipped basis h5 dataset.
        histrange_x, histrange_y : list or tuple of 2 floats or ints
            Optionally put custom bin ranges.
        no_pbar : bool
            Optionally do not include the progress bar for pdist generation.
        TODO: maybe also binsfromexpression?
        """
        # TODO: maybe change self.f to self.h5?
        self.h5 = h5
        self.f = h5py.File(h5, mode="r")
        if data_type is None:
            raise ValueError("Must input valid data_type: `evolution`, `average`, or `instant`")
        else:
            self.data_type = data_type
        self.p_units = p_units
        self.T = T
        self.weighted = weighted

        # TODO: Default pcoord for either dim
        # TODO: clean up and condense this name processing section
        # add auxdata prefix if not using pcoord and not using array input
        # doing it in two conditional blocks since numpy as warning with comparing array to string
        # this way it only does the string comparison if Xname is a string
        if isinstance(Xname, str):
            if Xname != "pcoord":
                Xname = "auxdata/" + Xname
        self.Xname = Xname
        # TODO: set this up as an arg to be able to process 3D+ arrays form aux
        # need to define the index if pcoord is 3D+ array, index is ndim - 1
        self.Xindex = Xindex

        # for 1D plots, but could be better (TODO)
        if Yname is not None and isinstance(Yname, str):
            # for common case of evolution with extra Yname input
            if Yname and data_type == "evolution":
                message = "\nDefaulting to evolution plot for --data-type, since you put a --Yname arg, " + \
                        "\nDid you mean to use --data-type of `average` or `instant`?"
                warn(message)
            # add auxdata prefix if not using pcoord and not using array input
            if Yname != "pcoord":
                Yname = "auxdata/" + Yname
            # before comparing X and Y, make sure they are both strings
            if isinstance(Xname, str):
                # for the common case where one plots pcoord/aux 0 and pcoord/aux 1
                if Xname == Yname and Xindex == 0 and Yindex == 0:
                    Yindex = 1
                    message = "\nSetting --Yindex to 1 (2nd dimension) since Xname/Yname " + \
                            "and Xindex/Yindex were the same."
                    warn(message)
        self.Yname = Yname
        self.Yindex = Yindex

        # for replacing the Z axis pdist with a dataset
        if Zname is not None and isinstance(Zname, str):
            # add auxdata prefix if not using pcoord and not using array input
            if Zname != "pcoord":
                Zname = "auxdata/" + Zname
        self.Zname = Zname
        self.Zindex = Zindex

        # XYZ save into new h5 file options
        self.H5save_out = H5save_out
        self.Xsave_name = Xsave_name
        self.Ysave_name = Ysave_name
        self.Zsave_name = Zsave_name
        if H5save_out is not None:
            shutil.copyfile(self.h5, H5save_out)
            self.H5save_out = h5py.File(H5save_out, "r+")
        
        # raw data processing function
        # TODO: allow for 2-3 functions as tuple input, right now one function only
        self.data_proc = data_proc

        # default to last
        if last_iter is not None:
            self.last_iter = last_iter
        elif last_iter is None:
            self.last_iter = self.f.attrs["west_current_iteration"] - 1

        if data_type == "instant":
            self.first_iter = self.last_iter
        else:
            self.first_iter = first_iter

        self.bins = bins

        # first make a list for each iteration weight array
        weights = []
        #for iter in range(self.first_iter, self.last_iter + 1):
        # have to make array start from iteration 1 to index well during weighting
        # but only for using skipping basis
        if skip_basis is None:
            weight_start = self.first_iter
        elif skip_basis:
            weight_start = 1
        for iter in range(weight_start, self.last_iter + 1):
            weights.append(self.f[f"iterations/iter_{iter:08d}/seg_index"]["weight"])
        # 1D array of variably shaped arrays
        self.weights = np.array(weights, dtype=object)

        # integer for the amount of frames saved (length) per tau (e.g. 101 for 100 ps tau)
        self.tau = self._get_data_array("pcoord", 0, self.first_iter).shape[1]

        # n_particles for each iteration
        self.n_particles = self.f["summary"]["n_particles"]#[self.first_iter-1:self.last_iter]

        # TODO: I wonder if both of these attributes are needed (total only used by reshape data array)
        #       I should note somewhere that data array must be for the same length/iters as the west.h5 file
        # the sum of n segments in all specified iterations and all iterations overall
        self.current_particles = np.sum(self.f["summary"]["n_particles"][self.first_iter-1:self.last_iter])
        # do not include the final (empty) iteration
        self.total_particles = np.sum(self.f["summary"]["n_particles"][:-1])

        self.skip_basis = skip_basis
        self.skip_basis_out = skip_basis_out
        self.n_bstates = self.f["ibstates/index"]["n_bstates"]
        self.histrange_x = histrange_x
        self.histrange_y = histrange_y
        self.no_pbar = no_pbar

    def _get_data_array(self, name, index, iteration, h5_create=None, h5_create_name=None):
        """
        Extract, index, and return the aux/data array of interest.
        Rows are segments, columns are frames until tau, depth is ndimensional datasets.

        Parameters
        ----------
        name : str
            Dataset name.
        index : int
            Dataset index.
        iteration : int
            WE iteration.
        h5_create : str
            Name of the h5 file to add the dataset to.
        h5_create_name : str
            Name of the dataset that is being placed into the h5 file.

        Returns
        -------
        data : ndarray
            Dataset of interest from the H5 file.
        """
        # if the user puts in an array object instead of a string dataset name
        if isinstance(name, np.ndarray):
            # first reshape 1d input raw data array into 3d array
            # currently, this is done during pdist method
            #data = self.reshape_total_data_array(name)

            # need to parse data for segments only in current iteration
            # segments are each row, but in input they are all concatenated
            n_segs_up_to_iter = np.sum(self.n_particles[self.first_iter-1:iteration-1])
            n_segs_including_iter = np.sum(self.n_particles[self.first_iter-1:iteration])
            data = name[n_segs_up_to_iter:n_segs_including_iter,:,:]
        # name should be a string for the h5 file dataset name
        elif isinstance(name, str):
            # this t/e block is to catch non-existent aux data names
            try:
                data = np.array(self.f[f"iterations/iter_{iteration:08d}/{name}"])
            except KeyError:
                message = f"{name} is not a valid object in the h5 file. \n" + \
                          f"Available datasets are: 'pcoord' "
                # this t/e block is to catch the case where there are no aux datasets at all
                try:
                    auxnames = list(self.f[f"iterations/iter_{self.first_iter:08d}/auxdata"])
                    message += f"and the following aux datasets {auxnames}"
                except KeyError:
                    message += "and no aux datasets were found"
                raise ValueError(message)
                
        else:
            raise ValueError("Xname Yname and Zname arguments must be either a string or an array.")

        # standardize the data dimensions to allow 3d indexing
        data = np.atleast_3d(data)

        # run data processing function on raw data if available
        # don't do this for input array data
        if self.data_proc is not None and not isinstance(name, np.ndarray):
            data = self.data_proc(data)

        # option to fill out new h5 file with dataset included here
        # using westpa style compression and scaleoffset 
        # could round to 4 decimal places with data=np.around(data, 4)
        if h5_create:
            h5_create.create_dataset(f"iterations/iter_{iteration:08d}/auxdata/{h5_create_name}", 
                                     data=data, compression=4, scaleoffset=6, chunks=True)

        return data[:,:,index]

    # this does add a little overhead at high iteration ranges
    # ~0.5s from 100i to 400i
    # alternatively, can put histrange_x and histrange_y args to skip this
    def _get_histrange(self, name, index):
        """
        Get the histrange considering the min/max of all iterations considered.

        Parameters
        ----------
        name : str
            Target auxillary data name for range calculation.
        index : int
            Target auxillary data index for range calculation.

        Returns
        -------
        histrange : tuple
            2 item list of min and max bin bounds for hist range of target aux data.
        """
        # set base histrange based on first iteration
        iter_data = self._get_data_array(name, index, self.first_iter)
        histrange = [np.amin(iter_data), np.amax(iter_data)]

        # loop and update to the max and min for all other iterations considered
        for iter in range(self.first_iter + 1, self.last_iter + 1):
            # get min and max for the iteration
            iter_data = self._get_data_array(name, index, iter)
            iter_min = np.amin(iter_data)
            iter_max = np.amax(iter_data)

            # update to get the largest possible range from all iterations
            if iter_min < histrange[0]:
                histrange[0] = iter_min
            if iter_max > histrange[1]:
                histrange[1] = iter_max

        return histrange

    def _normalize(self, hist):
        """
        Normalize or convert the probabilities.

        Parameters
        ----------
        hist : ndarray
            Array containing the histogram count values to be normalized.

        Returns
        -------
        hist : ndarray
            The hist array is normalized according to the p_units argument. 
        """
        # -lnP
        if self.p_units == "kT":
            hist = -np.log(hist / np.max(hist))
        # -RT*lnP
        elif self.p_units == "kcal":
            # Gas constant R = 1.9872 cal/K*mol or 0.0019872 kcal/K*mol
            hist = -0.0019872 * self.T * np.log(hist / np.max(hist))
        # raw probability
        elif self.p_units == "raw":
            hist = hist
        # raw normalized probability (P(x)/P(max))
        elif self.p_units == "raw_norm":
            hist = hist / np.max(hist)
        else:
            raise ValueError("Invalid p_units value, must be 'kT', 'kcal', 'raw', or 'raw_norm'.")
        return hist

    def _get_children_indices(self, parent):
        """
        For a (iter, seg) pair, look for and return all iter + 1 segment indices.

        Parameters
        ----------
        parent : tuple
            (iteration, segment).

        Returns
        -------
        children : tuple
            The indices of all child segments from the input parent segment.
        """
        children = []
        p_iter, p_seg = parent
        # for all parent_id values of the iter + 1 iteration
        for idx, seg in enumerate(
        self.f[f"iterations/iter_{p_iter+1:08d}/seg_index"]["parent_id"]):
            # match with input parent segment
            if seg == p_seg:
                # put all the parent segment children's indices into list
                children.append(idx)
                
        return children

    def _new_weights_from_skip_basis(self):
        """
        Make a new temp h5 file with zero weights for skipped basis state walkers.

        Returns
        -------
        self.weights : array
            Updated weight array with zero values for skipped basis states.
        """
        # TODO: doesn't work with --first-iter
        
        # copy of weights to edit
        new_weights = self.weights

        # setup a warning for h5 files that have incorrectly recorded bstate pcoords
        # this will all be based off of the first pcoord array (Z index 0)
        # correspondingly, the bstate_pcoord will be the first column
        bs_coords = self.f[f"ibstates/0/bstate_pcoord"]
        it1_coords = self.f[f"iterations/iter_00000001/pcoord"][:,0,0]
        # need to second element get the unique indices
        it1_unique_indices = np.unique(it1_coords, return_index=True)[1]
        # then sort to the original bstate ordering
        it1_unique_coords = np.array([it1_coords[index] \
                            for index in sorted(it1_unique_indices)])
        try:
            # make sure that traced unique pcoord elements match the basis state values
            if np.isclose(bs_coords[:,0], it1_unique_coords, rtol=1e-04) is False:
                message = f"The traced pcoord \n{it1_unique_coords} \ndoes not equal " + \
                          f"the basis state coordinates \n{bs_coords}"
                warn(message)
        except ValueError as e:
            message = f"{e}: Not all bstates may have been used in iteration 1."
            warn(message)

        # TODO: print bstate pcoords
        print(f"bstates: {bs_coords[:,0]}")
        print(f"bstates from pcoord: {it1_unique_coords}")
        #import sys ; sys.exit(0)

        # if the basis state binary is a 1 in skip_basis, use weight 0 
        #print("First run skip_basis processing from each initial segment: ")
        for basis, skip in enumerate(self.skip_basis):
            # essentially goes through all initial segments for each skipped basis
            if skip == 1:
                # loop of each initial pcoord value from iteration 1
                for it1_idx, it1_val in enumerate(it1_coords):
                    # so if the pcoord value matches the bstate value to be skipped
                    # needs to both be at the same precision
                    if np.isclose(it1_val, bs_coords[basis,0], rtol=1e-04):
                        # search forward to look for children of basis state 
                        # start at it1_idx, make weight zero 
                        new_weights[0][it1_idx] = 0

                        # list for parent_ids of the current segment skip basis lineage
                        skip_parents_c = [it1_idx]
                        # list for storing the indices to skip for the next iteration
                        skip_parents_n = []

                        # zero the next iteration's children until last_iter
                        for iter in tqdm(range(1, self.last_iter + 1), 
                                         desc="skip_basis", disable=self.no_pbar):
                            for idx in skip_parents_c:
                                # make zero for each child of skip_basis
                                new_weights[iter-1][idx] = 0
                                # then make new skip_parents tuple to loop for next iter
                                skip_parents_n += self._get_children_indices((iter, idx))

                            # make new empty list to store the iteration's skipped
                            skip_parents_c.clear()
                            skip_parents_c += skip_parents_n
                            skip_parents_n.clear()

        # TODO: prob can do better than these print statements
        print("pdist calculation: ")
        # write new weights into skip_basis_out h5 file
        if self.skip_basis_out is not None:
            shutil.copyfile(self.h5, self.skip_basis_out)
            h5_skip_basis = h5py.File(self.skip_basis_out, "r+")
            for idx, weight in enumerate(new_weights):
                h5_skip_basis[f"iterations/iter_{idx+1:08d}/seg_index"]["weight"] = weight
            
        # only return portion of weights requested by user
        return new_weights[self.first_iter-1:self.last_iter]

    # TODO: clean up and optimize
    def search_aux_xy_nn(self, val_x, val_y):
        """
        Originally adapted from code by Jeremy Leung.
        Tree search to find closest datapoint to input data value(s).

        # TODO: add step size for searching, right now gets the last frame

        Parameters
        ----------
        val_x : int or float
            X dataset value to search for.
        val_y : int or float
            Y dataset value to search for.
        """
        # iter is already known when searching evo data
        if self.data_type == "evolution":
            iter_num = int(val_y)
            
            # These are the auxillary coordinates you're looking for
            r1 = self._get_data_array(self.Xname, self.Xindex, iter_num)[:,-1]

            # phase 2: finding seg number

            # TODO: numpy array this
            small_array2 = []
            for j in range(0,len(r1)):
                small_array2.append([r1[j]])
            tree2 = KDTree(small_array2)

            # TODO: these can be multiple points, maybe can parse these and filter later
            d2, i2 = tree2.query([val_x],k=1)
            seg_num = int(i2)

        else:
            # phase 1: finding iteration number
            distances = []
            indices = []

            # change indices to number of iteration
            #for i in range(self.first_iter, self.last_iter + 1): 
            # always use iteration 1 to get full trace path
            for i in range(1, self.last_iter + 1): 

                # These are the auxillary coordinates you're looking for
                r1 = self._get_data_array(self.Xname, self.Xindex, i)[:,-1]
                r2 = self._get_data_array(self.Yname, self.Yindex, i)[:,-1]

                small_array = []
                for j in range(0,len(r1)):
                    small_array.append([r1[j],r2[j]])
                tree = KDTree(small_array)

                # Outputs are distance from neighbour (dd) and indices of output (ii)
                dd, ii = tree.query([val_x, val_y],k=1) 
                distances.append(dd) 
                indices.append(ii)

            minimum = np.argmin(distances)
            iter_num = int(minimum+1)

            # These are the auxillary coordinates you're looking for
            r1 = self._get_data_array(self.Xname, self.Xindex, iter_num)[:,-1]
            r2 = self._get_data_array(self.Yname, self.Yindex, iter_num)[:,-1]

            # phase 2: finding seg number

            # TODO: numpy array this
            small_array2 = []
            for j in range(0,len(r1)):
                small_array2.append([r1[j],r2[j]])
            tree2 = KDTree(small_array2)

            # TODO: these can be multiple points, maybe can parse these and filter later
            d2, i2 = tree2.query([val_x, val_y],k=1)
            seg_num = int(i2)

        #print("go to iter " + str(iter_num) + ", " + "and seg " + str(seg_num))
        print(f"Go to ITERATION: {iter_num} and SEGMENT: {seg_num}")
        return iter_num, seg_num

    ##################### TODO: update or organize this #############################
    def get_parents(self, walker_tuple):
        """
        Get parent of an input (iteration, walker).

        Parameters
        ----------
        walker_tuple : tuple
            (iteration, walker)

        Returns
        -------
        parent : iteration, walker
        """
        it, wlk = walker_tuple
        parent = self.f[f"iterations/iter_{it:08d}"]["seg_index"]["parent_id"][wlk]
        return it-1, parent

    def trace_walker(self, walker_tuple):
        """
        Get trace path of an input (iteration, walker).

        Parameters
        ----------
        walker_tuple : tuple
            (iteration, walker)

        Returns
        -------
        trace : list of tuples
            Tuples are (iteration, walker) traces.
        """
        # Unroll the tuple into iteration/walker 
        it, wlk = walker_tuple
        # Initialize our path
        path = [(it,wlk)]
        # And trace it
        while it > 1: 
            it, wlk = self.get_parents((it, wlk))
            path.append((it,wlk))
        return np.array(sorted(path, key=lambda x: x[0]))

    def get_coords(self, path, data_name, data_index):
        """
        Get a list of data coordinates for plotting traces.

        Parameters
        ----------
        path : list of tuples
            Tuples are (iteration, walker) traces.
        data_name : str
            Name of dataset.
        data_index : int
            Index of dataset.

        Returns
        -------
        coordinates : array
            Array of coordinates from the list of (iteration, walker) tuples.
        """
        # Initialize a list for the pcoords
        coords = []
        # Loop over the path and get the pcoords for each walker
        for it, wlk in path:
            coords.append(self._get_data_array(data_name, data_index, it)[wlk][::10])
        return np.array(coords)

    def plot_trace(self, walker_tuple, color="white", ax=None):
        """
        Plot trace.

        Parameters
        ----------
        walker_tuple : tuple
            (iteration, walker) start point to trace from.
        color : str
        ax : mpl axes object
        """
        # TODO: update/streamline this
        if ax is None:
            fig, ax = plt.subplots(figsize=(7,5))
        else:
            fig = plt.gcf()

        path = self.trace_walker(walker_tuple)
        # adjustments for plothist evolution of only aux_x data
        if self.data_type == "evolution":
            # split iterations up to provide y-values for each x-value (pcoord)
            aux = self.get_coords(path, self.Xname, self.Xindex)
            iters = np.arange(1, len(aux)+1)
            ax.plot(aux[:,0], iters, c="black", lw=2)
            ax.plot(aux[:,0], iters, c=color, lw=1)
            return

        # And pull aux_coords for the path calculated
        aux_x = self.get_coords(path, self.Xname, self.Xindex)
        aux_y = self.get_coords(path, self.Yname, self.Yindex)

        ax.plot(aux_x[:,0], aux_y[:,0], c="black", lw=2)
        ax.plot(aux_x[:,0], aux_y[:,0], c=color, lw=1)

    def w_succ(self):
        """
        Find and return all successfully recycled (iter, seg) pairs.
        TODO eventually can use this to plot pdist of succ only trajs
        note that I would have to norm by the overall pmax (not just succ pmax)
        Could have this be an optional feature.
        """
        succ = []
        for iter in range(self.last_iter):
            # if the new_weights group exists in the h5 file
            if f"iterations/iter_{iter:08d}/new_weights" in self.h5:
                prev_segs = self.f[f"iterations/iter_{iter:08d}/new_weights/index"]["prev_seg_id"]
                # append the previous iter and previous seg id recycled
                for seg in prev_segs:
                    succ.append((iter-1, seg))
        # TODO: order this by iter and seg vals? currently segs not sorted but is iter ordered
        return succ
    def succ_pdist(self):
        """
        TODO: Filter weights to be zero for all non successfull trajectories.
        Make an array of zero weights and fill out weights for succ trajs only.
        """
        pass
    ###############################################################################

    def aux_to_pdist_1d(self, iteration):
        """
        Take the auxiliary dataset for a single iteration and generate a weighted
        1D probability distribution. 

        Parameters
        ----------
        iteration : int
            Desired iteration to extract timeseries info from.

        Returns
        -------
        midpoints_x : ndarray
            Histogram midpoint bin values for target aux coordinate of dimension 0.
        midpoints_y : ndarray
            Optional histogram midpoint bin values for target aux coordinate of dimension 1.
        histogram : ndarray
            Raw histogram count values of each histogram bin. Can be later normalized as -lnP(x).
        """
        # return 1D aux data: 1D array for histogram and midpoint values
        aux = self._get_data_array(self.Xname, self.Xindex, iteration)

        # make an 1D array to fit the hist values based off of bin count
        histogram = np.zeros((self.bins[0]))
        for seg in range(0, aux.shape[0]):
            counts, bins = np.histogram(aux[seg], bins=self.bins[0], range=self.histrange_x)

            # selectively apply weights
            if self.weighted is True:
                # multiply counts vector by weight scalar from weight array
                counts = np.multiply(counts, self.weights[iteration - self.first_iter][seg])

            # add all of the weighted walkers to total array for the 
            # resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins[:-1] + bins[1:]) / 2
        
        # TODO: also save as instance attributes
        return midpoints_x, histogram

    def aux_to_pdist_2d(self, iteration):
        """
        Take the auxiliary dataset for a single iteration and generate a weighted
        2D probability distribution. 

        Parameters
        ----------
        iteration : int
            Desired iteration to extract timeseries info from.

        Returns
        -------
        midpoints_x : ndarray
            Histogram midpoint bin values for target aux coordinate of dimension 0.
        midpoints_y : ndarray
            Optional histogram midpoint bin values for target aux coordinate of dimension 1.
        histogram : ndarray
            Raw histogram count values of each histogram bin. Can be later normalized as -lnP(x).
        """
        # 2D instant histogram and midpoint values for a single specified WE iteration
        X = self._get_data_array(self.Xname, self.Xindex, iteration)
        Y = self._get_data_array(self.Yname, self.Yindex, iteration)

        # 2D array to store hist counts for each timepoint in both dimensions
        histogram = np.zeros(self.bins)
        for seg in range(0, X.shape[0]):
            counts, bins_x, bins_y = np.histogram2d(X[seg], Y[seg], 
                                                    bins=self.bins, 
                                                    range=[self.histrange_x, 
                                                           self.histrange_y]
                                                    )

            if self.weighted is True:
                # multiply counts vector by weight scalar from weight array
                counts = np.multiply(counts, self.weights[iteration - self.first_iter][seg])

            # add all of the weighted walkers to total array for 
            # the resulting linear combination
            histogram = np.add(histogram, counts)

        # get bin midpoints
        midpoints_x = (bins_x[:-1] + bins_x[1:]) / 2
        midpoints_y = (bins_y[:-1] + bins_y[1:]) / 2

        # TODO: save these as instance attributes
        # this will make it easier to save into a text pdist file later

        # save midpoints and transposed histogram (corrected for plotting)
        return midpoints_x, midpoints_y, histogram.T

    def evolution_pdist(self):
        """
        Returns the pdist for 1 coordinate for the range iterations specified.

        Returns
        -------
        x, y, norm_hist : arrays
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # make array to store hist (-lnP) values for n iterations of X
        evolution_x = np.zeros((self.last_iter - self.first_iter + 1, self.bins[0]))
        positions_x = np.zeros((self.last_iter - self.first_iter + 1, self.bins[0]))

        for iter in tqdm(range(self.first_iter, self.last_iter + 1), 
                         desc="Evolution", disable=self.no_pbar):
            # account for first_iter arg for array indexing
            iter_index = iter - self.first_iter + 1
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter)
            evolution_x[iter_index - 1] = counts_total_x
            positions_x[iter_index - 1] = center_x

        # 2D evolution plot of X (Y not used if provided) per iteration        
        evolution_x = self._normalize(evolution_x)

        # bin positions along aux x, WE iteration numbers, z data
        return positions_x, np.arange(self.first_iter, self.last_iter + 1, 1), evolution_x

    # TODO: maybe don't need individual functions, maybe can handle in main
    def instant_pdist_1d(self):
        """
        Returns the x and y pdist datasets for a single iteration.

        Returns
        -------
        Xdata, y : arrays
            x (dataset) and y (pdist) axis values
        """
        center, counts_total = self.aux_to_pdist_1d(self.last_iter)
        counts_total = self._normalize(counts_total)
        return center, counts_total

    def instant_pdist_2d(self):
        """
        Returns the xyz pdist datasets for a single iteration.

        Returns
        -------
        x, y, norm_hist : arrays
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        center_x, center_y, counts_total = self.aux_to_pdist_2d(self.last_iter)
        counts_total = self._normalize(counts_total)
        return center_x, center_y, counts_total

    def instant_datasets_3d(self):
        """
        Unique case where `Zname` is specified and the XYZ datasets are returned.
        For single iteration.

        Returns
        -------
        X, Y, Z : arrays 
            Raw data for each named coordinate.
        """
        X = self._get_data_array(self.Xname, self.Xindex, self.last_iter)
        # for the case where Zname is specified but not Yname
        if self.Yname is None:
            warn("`Zname` is defined but not `Yname`, using Yname=`pcoord`")
            Y = self._get_data_array("pcoord", self.Yindex, self.last_iter)
        else:
            Y = self._get_data_array(self.Yname, self.Yindex, self.last_iter)
        Z = self._get_data_array(self.Zname, self.Zindex, self.last_iter)

        return X, Y, Z

    def average_pdist_1d(self):
        """
        1 dataset: average pdist for a range of iterations.

        Returns
        -------
        x, y
            x and y axis values, and if using Y or evolution (with only X), 
            also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # make array to store hist (-lnP) values for n iterations of X
        evolution_x = np.zeros((self.last_iter, self.bins[0]))
        positions_x = np.zeros((self.last_iter, self.bins[0]))

        for iter in tqdm(range(self.first_iter, self.last_iter + 1), 
                         desc="Average 1D", disable=self.no_pbar):
            # generate evolution x data
            center_x, counts_total_x = self.aux_to_pdist_1d(iter)
            evolution_x[iter - 1] = counts_total_x
            positions_x[iter - 1] = center_x

        # summation of counts for all iterations : then normalize
        col_avg_x = [np.sum(col[col != np.isinf]) for col in evolution_x.T]
        col_avg_x = self._normalize(col_avg_x)

        # 1D average plot data for X
        return center_x, col_avg_x

    def average_pdist_2d(self):
        """
        2 datasets: average pdist for a range of iterations.

        Returns
        -------
        x, y, norm_hist
            x and y axis values, and if using Y or evolution (with only X), also returns norm_hist.
            norm_hist is a 2-D matrix of the normalized histogram values.
        """
        # empty array for 2D pdist
        average_xy = np.zeros(self.bins)

        # 2D avg pdist data generation
        for iter in tqdm(range(self.first_iter, self.last_iter + 1), 
                         desc="Average 2D", disable=self.no_pbar):
            center_x, center_y, counts_total_xy = self.aux_to_pdist_2d(iter)
            average_xy = np.add(average_xy, counts_total_xy)

        average_xy = self._normalize(average_xy)
        return center_x, center_y, average_xy

    def average_datasets_3d(self, interval=1):
        """
        Unique case where `Zname` is specified and the XYZ datasets are returned.
        Averaged over the iteration range.
        
        Returns
        -------
        X, Y, Z : arrays 
            Raw data for each named coordinate.
        """
        if self.Yname is None:
            warn("`Zname` is defined but not `Yname`, using Yname=`pcoord`")
            self.Yname = "pcoord"

        # arrays to be filled with values from each iteration
        # rows are for all segments, columns are each segment datapoint
        X = np.zeros((self.current_particles, self.tau))
        Y = np.zeros((self.current_particles, self.tau))
        Z = np.zeros((self.current_particles, self.tau))

        # loop each iteration
        seg_start = 0
        for iter in tqdm(range(self.first_iter, self.last_iter + 1), 
                         desc="Average 3D", disable=self.no_pbar):
            # then go through and add all segments/walkers in the iteration
            X[seg_start:seg_start + self.n_particles[iter - 1]] = \
                self._get_data_array(self.Xname, self.Xindex, iter)
            Y[seg_start:seg_start + self.n_particles[iter - 1]] = \
                self._get_data_array(self.Yname, self.Yindex, iter)
            Z[seg_start:seg_start + self.n_particles[iter - 1]] = \
                self._get_data_array(self.Zname, self.Zindex, iter)

            # keeps track of position in the seg_total length based arrays
            seg_start += self.n_particles[iter - 1]

        # 3D average datasets using all available data (can more managable with interval)
        return X[::interval], Y[::interval], Z[::interval]

    def get_all_weights(self):
        """
        Returns an 1D array of the weight for every frame of each tau 
        for all segments of all iterations specified.

        Returns
        -------
        weights_expanded : array
        """
        # weights per seg of each iter, but need for each frame
        weights_1d = np.concatenate(self.weights)

        # need each weight value to be repeated for each tau (e.g. 100 + 1) 
        # will be same shape as X or Y made into 1d shape
        weights_expanded = np.zeros(self.tau * self.current_particles)

        # loop over all ps intervals up to tau in each segment
        weight_index = 0
        for seg in weights_1d:
            # TODO: can I do this without the unused loop?
            for frame in range(self.tau):
                weights_expanded[weight_index] = seg
                weight_index += 1

        return weights_expanded

    # TODO: option for data and weight output for a single iteration (iteration=None)
    # wait, isn't that already available in _get_data_array?
    def get_total_data_array(self, name, index=0, interval=1, reshape=True):
        """
        Loop through all iterations specified and get a 1d raw data array.
        # TODO: this could be organized better with my other methods
        maybe I can separate the helper functions into another class
        for extracting and moving data around, this pdist class could
        be used strictly for making pdists from a nice and standard data
        array input that is handled by the H5_Processing class

        Parameters
        ----------
        name : str
            Name of data from h5 file such as `pcoord` or an aux dataset.
        index : int
            Index of the data from h5 file.
        interval : int
            If more sparse data is needed for efficiency.
        reshape : bool
            Option to reshape into 1d array instead of each seg for all tau values.

        Returns
        -------
        data : 1d array
            Raw (unweighted) data array for the name specified.
        """
        data = np.zeros((self.current_particles, self.tau))
        # account for non-pcoord input strings
        if name != "pcoord":
            name = "auxdata/" + name
    
        # loop each iteration
        seg_start = 0
        for iter in tqdm(range(self.first_iter, self.last_iter + 1), 
                         desc="Getting Data Array", disable=self.no_pbar):
            # then go through and add all segments/walkers in the iteration
            data[seg_start : seg_start + self.n_particles[iter - 1]] = \
                self._get_data_array(name, index, iter)
            
            # keeps track of position in the seg_total length based arrays
            seg_start += self.n_particles[iter - 1]

        if reshape:
            return data[::interval].reshape(-1,1)
        else:
            return data[::interval]

    def reshape_total_data_array(self, array):
        """
        Take an input 1d array of the data values at every segment for each
        iteration, and reshape them to make pdists.

        Parameters
        ----------
        array : 1d array
            Data values at every segment for each iteration.

        Returns
        -------
        array : ndarray
            Now rows = segments, columns = frame until tau, depth = data dimensions.
        """
        # try except block for input from data array
        # which can be the correct shape if pulled from westpa (e.g. 100 ps + 1)
        # or if from agg MD sim, will just be (e.g. 100 ps)
        # also adding -1 in z dim for extra depth dimension compatibility

        # TODO: change total particles to iteration range to be able to use iter args with data arrays

        try:
            array = array.reshape(self.total_particles, self.tau, -1)
        # e.g. ValueError: cannot reshape array of size 303000 into shape (3000,100,newaxis)
        except ValueError as e:
            array = array.reshape(self.total_particles, self.tau - 1, -1)
            message = "\nYou may be using an input data array which did not include the rst file datapoints. " + \
                      "\nThis may be fine, but note that you shouldn't create a new H5 file using this array."
            warn(e + message)
            # TODO: does this work?
            # the case where the array does not have rst data included
            # put the new first column as the first value of each row (segment)
            # TODO: this is a temp hack for the no rst shape data
            # noting that both arrays must have same ndims for hstack
            #print(f"original shape: {data.shape}")
            #print(f"to stack shape: {data[:,0,:]}")
            array = np.hstack((np.atleast_3d(array[:,0,:]), array)) # TODO: test this
            #print(f"new shape: {data.shape}")

        # TODO: the above works to solve the shape issue but if I wanted to fill out a new dataset in
        # the h5 file, it would be missing the first value, which links walkers.
        # maybe I can use the parent IDs to link it manually, but note I would have to
        # go through and parse by my self.n_particles array to separate iterations.
        # put conditional if shape[1] = tau vs tau - 1 for creating dataset (to add parent data point)
        # note that the first iteration I need to pull from somewhere else? It's calculated from the
        # original bstate file

        # Note, if the user includes the rst files like WESTPA does, it should look and process fine

        return array
        
    def pdist(self):
        """
        Main public method with pdist generation controls.
        """ 
        # option to zero weight out specific basis states
        if self.skip_basis is not None:
            try: 
                self.weights = self._new_weights_from_skip_basis()
            # if the wrong amount of args are input and != n_bstates
            except IndexError as e:
                message = f"IndexError ({e}) for bstate input ({self.skip_basis}): " + \
                          f"Did you use the correct amount of bstates {self.n_bstates}?"
                warn(message)

        # reshape 1d input raw data array (if given) into 3d array
        if isinstance(self.Xname, np.ndarray):
            self.Xname = self.reshape_total_data_array(self.Xname)
        if isinstance(self.Yname, np.ndarray):
            self.Yname = self.reshape_total_data_array(self.Yname)
        if isinstance(self.Zname, np.ndarray):
            self.Zname = self.reshape_total_data_array(self.Zname)

        # TODO: could make this it's own method
        # if requested, save out a new H5 file with the input data array in new aux name
        if self.H5save_out is not None:
            for iter in range(self.first_iter, self.last_iter + 1):
                if self.Xsave_name:
                    self._get_data_array(self.Xname, self.Xindex, iter, self.H5save_out, self.Xsave_name)
                if self.Ysave_name:
                    self._get_data_array(self.Yname, self.Yindex, iter, self.H5save_out, self.Ysave_name)
                if self.Zsave_name:
                    self._get_data_array(self.Zname, self.Zindex, iter, self.H5save_out, self.Zsave_name)

        # TODO: need to consolidate the Y 2d vs 1d stuff somehow

        # TODO: if I can get rid of this or optimize it, I can then use the 
            # original methods of each pdist by themselves
        # TODO: only if histrange is None
        if self.histrange_x is None:
            # get the optimal histrange
            self.histrange_x = self._get_histrange(self.Xname, self.Xindex)
        # if using 2D pdist
        # TODO: needs to handle array input or None input
        if isinstance(self.Yname, (str, np.ndarray)) and self.histrange_y is None:
            self.histrange_y = self._get_histrange(self.Yname, self.Yindex)

        # TODO: need a better way to always return XYZ (currently using ones)
        if self.data_type == "evolution":
            return self.evolution_pdist()
        elif self.data_type == "instant":
            if self.Yname and self.Zname:
                return self.instant_datasets_3d()
            elif self.Yname:
                return self.instant_pdist_2d()
            else:
                X, Y = self.instant_pdist_1d()
                return X, Y, np.ones((self.first_iter, self.last_iter))
        elif self.data_type == "average":
            # attemts to say, if not None, but has to be compatible with str and arrays
            if isinstance(self.Yname, (str, np.ndarray)) and isinstance(self.Zname, (str, np.ndarray)):
                return self.average_datasets_3d()
            elif isinstance(self.Yname, (str, np.ndarray)):
                return self.average_pdist_2d()
            else:
                X, Y = self.average_pdist_1d()
                return X, Y, np.ones((self.first_iter, self.last_iter))

#if __name__ == "__main__":
    # total_array_out = np.loadtxt("p53_X_array.txt")
    # original_array = np.loadtxt("p53_X_array_noreshape.txt")
    
    # h5 = H5_Pdist("data/p53.h5", data_type="evolution")
    # TODO: test Zname with data_array