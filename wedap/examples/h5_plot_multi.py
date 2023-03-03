
from h5_plot import *
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.style.use("default.mplstyle")

# see AJD-autocorr plot: can use plt.gcf() (get current figure) in plotting function for multi fig subplots
    
fig, ax = plt.subplots(ncols=4, sharey=True, figsize=(14,4.5), gridspec_kw={'width_ratios' : [20, 20, 20, 1.5]})

for val, item in enumerate([50, 100, 150]):
    data_options = {"data_type" : "average",
                    "p_max" : 20,
                    "p_units" : "kT",
                    "last_iter" : item,
                    "bins" : 100,
                    }

    #X, Y, Z = pdist_to_normhist("2kod_v02/wcrawl/west_i200_crawled.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
    X, Y, Z = pdist_to_normhist("1a43_v02/wcrawl/west_i200_crawled.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)
    #X, Y, Z = pdist_to_normhist("1a43_v01/wcrawl/west_i150_crawled.h5", "1_75_39_c2", "fit_m1_rms_heavy_m2", **data_options)

    levels = np.arange(0, data_options["p_max"] + 1, 1)
    lines = ax[val].contour(X, Y, Z, levels=levels, colors="black", linewidths=1)
    plot = ax[val].contourf(X, Y, Z, levels=levels, cmap="gnuplot_r")    
    ax[val].set(ylim=(0,20), xlim=(0,90))#, title=f"Iter: {item}")
    ax[val].set_title(f"Iter: {item}", fontweight="bold")
    ax[val].axhline(y=7.5, color="cornflowerblue", linewidth=0.75)
    ax[val].axvline(x=45, color="cornflowerblue", linewidth=0.75)
    #ax[val].spines['top'].set_visible(False)
    #ax[val].spines['right'].set_visible(False)
    #ax[val].xaxis.set_minor_locator(MultipleLocator(5))
    #ax[val].yaxis.set_minor_locator(MultipleLocator(1))
    if val == 0:
        ax[val].set(ylabel=r"M2 RMSD ($\AA$)")


cbar = fig.colorbar(plot, cax=ax[3])
fig.suptitle("CA CTD Xtal (PDB:1A43)")
fig.text(0.5, 0.0, "Helical Angle (degrees)", ha='center', va='bottom')
#fig.text(0.0, 0.48, r"M2 RMSD ($\AA$)", va='center', rotation='vertical', ha='left')

cbar.set_label(r"$\Delta F(\vec{x})\,/\,kT$" + "\n" + r"$\left[-\ln\,P(x)\right]$")
#cbar.set_label(r"$\it{-RT}$ ln $\it{P}$ (kcal mol$^{-1}$)")
fig.tight_layout()
fig.savefig("figures/2kod_v01_150i_c2_m2_avg_evo_states_large.png", dpi=300, transparent=True)
#plt.show()



