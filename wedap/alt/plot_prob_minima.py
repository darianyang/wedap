"""
Need to import get prob of aux from h5 function, then can plot prob minima per iteration.
"""

# import wedap
# import numpy as np
# from numpy import inf
# import pandas as pd
# import matplotlib.pyplot as plt

# # Suppress divide-by-zero in log
# np.seterr(divide='ignore', invalid='ignore')

# def calc_minimum_free(midpoints, histogram):
#     """
#     Returns minimum pcoord value for the target iteration.
#     """
#     # here I could make 2 row ndarray instead
#     pdist = dict(zip(histogram, midpoints))

#     # get pcoord value at minimum probability / free energy
#     return pdist[np.min(histogram)]

# def plot_prob_minima(aux, fraction=None):
#     minima = []
#     for i in range(1,134):
#         center, counts_total = wedap.aux_to_pdist("1a43_v01/west_i150.h5", i, aux)
#         if fraction:
#             center = np.divide(center, fraction)
#         minima.append(calc_minimum_free(center, counts_total))
#     plt.plot(minima, alpha=1, linewidth=1.5, label=aux)

### RMS Aux Data ###
# rmss = ["RMS_Heavy", "RMS_Backbone", "RMS_Dimer_Int", "RMS_Key_Int", "RMS_Mono1", "RMS_Mono2"]
# for i in rmss:
#     plot_prob_minima(i)

### SASA Aux Data ###
# sasa = ["Mono1_SASA", "Mono2_SASA"]
# for i in sasa:
#     plot_prob_minima(i)

### RoG Aux Data ###
#plot_prob_minima("RoG")

### Contacts Aux Data ###
# TODO: find out the value from cpptraj westpa aux calc (or recalc) - also for NNC
# there are 232 inter NC, 6194 intra NC from ???
# also used 66 inter NC, 5242 intra NC: from 'm01/1us_nc_4.5A/m01_2kod_nc_res_pairs_1us_inter_4.5A.dat'
# There are 66 inter NC and 664 inter NNC, 12640 intra NNC
# contacts = {"Num_Inter_NC":66, "Num_Inter_NNC":664, "Num_Intra_NC":5242, "Num_Intra_NNC":12640}
# for key, value in contacts.items():
#     plot_prob_minima(key, fraction=value)

# plt.ylim(0,1)

### universal plotting and formatting ###
# plt.xlabel("WE Iteration")
# plt.ylabel(r'Minimum $\Delta F(x)\,/\,kT$ $\left[-\ln\,P(x)\right]$ ($\AA$)')
# plt.legend(loc=2)
# plt.show()
#plt.savefig("figures/west_min_all_contacts_i133.png", dpi=300)
