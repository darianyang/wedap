#!/usr/bin/python

from scipy.stats import hmean
import random
import numpy as np
from tqdm.auto import tqdm

################################################
############## DEFINING CI FUNCTION ############
################################################

def get_CI(List, Repeat):
    """
    Standard bootstrapping CIs.
    """
    if len(set(List)) == 1 :  	# NO CI if List has only identical elements
        return [0,0]

    else :
        AllMeans = []		# List of data(!) sample means for every bootstrap iteration
        N = len(List)                # Number of data points

        for i in range(Repeat) :	# Repeated bootstrap iterations
            CurrList = []		# Sample list
            for j in range(N) :
                CurrList.append(random.choice(List))
            AllMeans.append(np.average(CurrList))

    perc_min  = np.percentile(AllMeans,2.5)	# Minimum percentile defined over list of means 
    perc_max  = np.percentile(AllMeans,97.5) 	# Maximum percentile defined over list of means

    return [perc_min, perc_max]	# Confidence Interval is defined by min/max percentiles of sampling means


################################################
############## DEFINING CR FUNCTION ############
################################################

def get_CR_bm(List, Repeat):
    """
    Original bayesian bootstrapping function from BM.
    """
    if len(set(List)) == 1 :  	# NO CR if List has only identical elements
        return [0,0]

    else :
        AllMeans = []		# List of model(!) sample means for every bootstrap iteration
        N = len(List)		# Number of data points

        for i in range(Repeat) :	# Repeated bootstrap iterations
            Rands = [0]		# Following Rubin et al. to get data probabilities from Dirichlet distrib.
            CurrAvg = 0
            for j in range(N-1) :
                Rands.append(random.random()) 
            Rands.append(1)
            Rands.sort()
            P=np.diff(Rands)	# List of random numbers that add to 1 and are used as data probabilities
            for j in range(N) :
                CurrAvg += P[j]*List[j]	# Sample mean
            AllMeans.append(CurrAvg)		
    
    AllMeans.sort()
    TotalProb = len(AllMeans)
    CumulProb = 0
    perc_min  = 0
    perc_max  = 0
    for m in AllMeans :	# Iterating through sorted means, identifying that mean at which a certain percentile of probs is reached
        CumulProb += 1
        if (CumulProb > 0.025*TotalProb) and (perc_min == 0) :
            perc_min = m   
        if (CumulProb > 0.975*TotalProb) and (perc_max == 0):
            perc_max = m

    return [perc_min, perc_max]		# Credibility Region is defined by min/max percentiles of sampling means 

#####################################################################
################## Updated functions using numpy ####################
#####################################################################

def get_CR_single(rates, repeat):
    """
    Get a set of min and max credibility regions (CRs) from bayesian
    bootstrapping for a 1d array of n_replicates at a single timepoint.

    Parameters
    ----------
    rates : 1darray
        Rates of each replicate at a single timepoint.
    repeat : int
        n-fold bayesian bootstrapping.
    statistic : function
        Statistic to calculate for input data. Defauly np.mean.

    Returns
    -------
    CRs : 1darray (min CR | max CR)
        2 item array of calculated min and max CRs for the input rates array.
    """
    # check if all elements in the rates array are identical
    if np.unique(rates).size == 1:
        return np.array([0, 0])

    else:
        all_stats = np.zeros(repeat)
        n = len(rates)

        # n-fold (n repeat) bayesian bootstrapping
        for i in range(repeat):
            # get data probabilities from Dirichlet distrib.
            rands = np.random.dirichlet(np.ones(n))
            # calculate the sample mean using dot product
            all_stats[i] = np.dot(rands, rates)

        # identifying the sorted mean at which a certain percentile of probs is reached
        all_stats.sort()
        total_prob = repeat
        cumul_prob = np.arange(1, total_prob + 1)
        # calculate the percentile values
        perc_min = all_stats[np.argmax(cumul_prob > 0.025 * total_prob)]
        perc_max = all_stats[np.argmax(cumul_prob > 0.975 * total_prob)]

        return np.array([perc_min, perc_max])

def get_CR_multi(rates_multi, repeat):
    """
    Calculate the credibility regions of multiple timepoints.
    Input `rates_multi` array should have n_replicate rows and
    n_timepoints columns.

    Parameters
    ----------
    rates_multi : 2darray
        Rates for multiple replicates at multiple timepoints.
    repeats : int
        n-fold bayesian bootstrapping.
    
    Returns
    -------
    CRs : 2darray
        Calculated min and max CRs for n_replicates at each timepoint.
        n_timepoints rows and 2 columns (min CR and max CR).
    """
    # CRs will be 2 columns (min and max) and n_frames rows
    CRs = np.zeros((rates_multi.shape[1], 2))
    # loop each timepoint, so each set of n_replicate rates, must transpose
    for i, rep_rates in enumerate(tqdm(rates_multi.T)):
        CRs[i,:] = get_CR_single(rep_rates, repeat)
    return CRs

def bayboot(X, statistic=None, n_replications=10000, resample_size=100, alpha=0.05):
    """
    Adapted from bayesian_bootstrapping package.
    Simulate the posterior distribution of the given statistic.
    The highest-density interval containing a (1-alpha) fraction of the posterior samples.

    Parameters
    ----------
    X : array
        The observed data.

    statistic : function
        A function of the data to use in simulation (Function mapping array-like to number).
        If None, uses mean.

    n_replications : int
        The number of bootstrap replications to perform (positive integer).

    resample_size : int
        The size of the dataset in each replication.

    alpha : float 
        The total size of the tails (Float between 0 and 1).

    Returns
    -------
    Left and right interval bounds (tuple)
    """
    if isinstance(X, list):
        X = np.array(X)

    # default to mean
    if statistic is None:
        weights = np.random.default_rng().dirichlet(np.ones(len(X)), n_replications)
        samples = np.dot(X, weights.T)
    
    # use custom statistic
    else:
        samples = []
        rng = np.random.default_rng()
        weights = rng.dirichlet([1] * len(X), n_replications)
        for w in weights:
            sample_index = rng.choice(range(len(X)), p=w, size=resample_size)
            resample_X = X[sample_index]
            s = statistic(resample_X)
            samples.append(s)

    # get highest density interval
    samples_sorted = sorted(samples)
    window_size = int(len(samples) - round(len(samples) * alpha))
    smallest_window = (None, None)
    smallest_window_length = float("inf")
    for i in range(len(samples_sorted) - window_size):
        window = samples_sorted[i + window_size - 1], samples_sorted[i]
        window_length = samples_sorted[i + window_size - 1] - samples_sorted[i]
        if window_length < smallest_window_length:
            smallest_window_length = window_length
            smallest_window = window
    return smallest_window[1], smallest_window[0]

def bayboot_multi(rates_multi, statistic=None, repeat=10000):
    """
    Calculate the credibility regions of multiple timepoints.
    Input `rates_multi` array should have n_replicate rows and
    n_timepoints columns.

    Parameters
    ----------
    rates_multi : 2darray
        Rates for multiple replicates at multiple timepoints.
    repeat : int
        n-fold bayesian bootstrapping.
    
    Returns
    -------
    CRs : 2darray
        Calculated min and max CRs for n_replicates at each timepoint.
        n_timepoints rows and 2 columns (min CR and max CR).
    """
    if isinstance(rates_multi, list):
        rates_multi = np.array(rates_multi)
    # CRs will be 2 columns (min and max) and n_frames rows
    CRs = np.zeros((rates_multi.shape[1], 2))
    # loop each timepoint, so each set of n_replicate rates, must transpose
    for i, rep_rates in enumerate(tqdm(rates_multi.T)):
        CRs[i,:] = bayboot(rep_rates, statistic=statistic, n_replications=repeat)
    return CRs

if __name__ == "__main__":
    # time the run
    import timeit
    start = timeit.default_timer()

    # check the updated numpy version works
    rates = [0.02, 0.005, 0.033]
    #rates = np.random.sample(500)
    print(get_CR_bm(rates, 10000))

    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"BM: Executed in {execution_time:04f} seconds\n")

    start = timeit.default_timer()
    print(list(get_CR_single(rates, 10000)))
    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"BM NP: Executed in {execution_time:04f} seconds\n")

    # test with bayesian-bootstrapping package
    from bayesian_bootstrap import mean, highest_density_interval
    start = timeit.default_timer()
    posterior_samples = mean(rates, 10000)
    print(highest_density_interval(posterior_samples))
    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"BB MEAN: Executed in {execution_time:04f} seconds\n")

    # my version of bayesian-bootstrapping package
    start = timeit.default_timer()
    print(bayboot(rates, statistic=None, resample_size=500))
    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"BB DTY MEAN: Executed in {execution_time:04f} seconds\n")

    # mean vs hmean
    from bayesian_bootstrap import bayesian_bootstrap

    start = timeit.default_timer()
    posterior_samples = bayesian_bootstrap(rates, np.mean, 10000, 100)
    print(highest_density_interval(posterior_samples))
    stop = timeit.default_timer()
    execution_time = stop - start
    print(f"BB: Executed in {execution_time:04f} seconds")

    # posterior_samples = bayesian_bootstrap(rates, hmean, 10000, 100)
    # print(highest_density_interval(posterior_samples))


    # how to make compatible with 2D?
    # a = np.random.sample((5,5))
    # bayboot(a)
