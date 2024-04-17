from bootstrap import *
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import hmean
from scipy.stats.mstats import gmean

n_runs = 4

# grab rate column (1) only
rates = [np.loadtxt(f"rate_0{i}.dat")[:,1] for i in range(2, n_runs + 1)]
# array of the 5 rate arrays
rates = np.array(rates)

# min non-zero value
min_val = np.amin(rates[np.nonzero(rates)])
# replace all zeros with smallest value measured
rates[rates == 0] = min_val

# get mean array of n replicates at each timepoint
means = np.average(rates, axis=0)
# calculate CRs at each timepoint
CRs = bayboot_multi(rates, repeat=100)
plt.plot(means, label="ameans")
plt.fill_between([i for i in range(0,rates.shape[1])], CRs[:,0], CRs[:,1], alpha=0.2)

# gmeans
means = gmean(rates)
CRs = bayboot_multi(rates, gmean, 100)
plt.plot(means, label="gmeans")
plt.fill_between([i for i in range(0,rates.shape[1])], CRs[:,0], CRs[:,1], alpha=0.2)

# hmeans
means = hmean(rates)
CRs = bayboot_multi(rates, hmean, 100)
plt.plot(means, label="hmeans")
plt.fill_between([i for i in range(0,rates.shape[1])], CRs[:,0], CRs[:,1], alpha=0.2)

# plotting
plt.yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])
plt.legend()
plt.savefig("comparison_ntr.png", dpi=300, transparent=False)
#plt.show()