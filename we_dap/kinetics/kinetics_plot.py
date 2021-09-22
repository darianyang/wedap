"""
Plot the output of WESTPA kinetics tools: w_direct -> direct.h5
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt

def kinetics_proc(h5, plot_type, step_iter=1, first_iter=1, last_iter=None, state=1, tau=100):
    """
    Parameters
    ----------
    h5 : str
        Path to kinetics h5 file (e.g. direct.h5).
    plot_type : str
        Can be 'flux', or 'rate'.
    step_iter : int
        The step size of the iterations to average. Should be based on correlation time.
    first_iter : int
        Starts at iteration 1 by default.
    last_iter : int
        Uses the last completed iteration in h5 file by default.
    state : int
        Column of dataset to process: default target state 1 (flux 0 -> 1).
    tau : int
        Tau value for WE iterations, in ns.
    """
    dataset = np.array(h5py.File(h5, mode="r")["total_fluxes"])[first_iter::step_iter, state]
    if last_iter:
        dataset = dataset[:int(last_iter / step_iter)]

    # molcular time for X axis
    mol_time = [i * tau * step_iter for i in range(0, len(dataset))]

    # cum avg of flux for Y axis (functionalize TODO)
    # cum_sum = []
    # iter = 0
    # curr_sum = 0
    # while iter <= (len(dataset) - 1) * step_iter:
    #     for i in range(iter, iter + step_iter):
    #         if i < len(dataset):
    #             curr_sum += dataset[i]
    #         else:
    #             continue
    #     cum_sum.append(curr_sum / (iter + step_iter))
    #     #print(curr_sum / (iter + step_iter))
    #     iter += step_iter
    # print(cum_sum)

    fig, ax = plt.subplots()

    if plot_type == "flux":
        ax.plot(mol_time, dataset)
        ax.set(xlabel=r"Molecular Time (ns)", ylabel="Mean Flux")
        plt.show()


    # note: rate_evolution = cumulatively averaged fluxes, so from iter 1 to iter X
    elif plot_type == "rate":
        # rates = dataset ** -1
        # #np.divide(dataset, tau * 10 ** -9)
        # for now, just use from h5, TODO: later calc direct from total fluxes
        rates = np.array(h5py.File(h5, mode="r")["rate_evolution"])[first_iter:, state]
        rates = [i[0][2] for i in rates]

        avg_rates = []
        iter = 0
        rate_sum = 0
        while iter <= len(rates):
            #rate_sum = 0
            for i in range(iter, iter + step_iter):
                if i < len(rates):
                    rate_sum += rates[i]
                else:
                    continue
            avg_rates.append(rate_sum / (iter + step_iter))
            iter += step_iter
        print(avg_rates)
        # divide by tau in seconds to get rate constant 
        rate_const = np.divide(avg_rates, tau * 10 ** -9)

        ax.plot(mol_time, rate_const)
        ax.set(xlabel=r"Molecular Time (ns)", ylabel=r"Rate Constant ($s^{-1}$)")
        plt.show()

    else:
        raise ValueError(f"plot_type of {plot_type} invalid: must be 'flux' or 'rate'.")

#kinetics_proc("2kod_v02/kinetics/C2_M2_1/direct.h5", plot_type="rate", step_iter=20, last_iter=None, tau=0.1)

kinetics_proc("1a43_v02/kinetics/C2_M2M1-Dist_1/direct.h5", plot_type="flux", step_iter=20, last_iter=None, tau=0.1)

