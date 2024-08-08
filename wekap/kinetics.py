"""
Plot the fluxes and rates from direct.h5 files.

list(h5):
['arrivals', 'avg_color_probs', 'avg_conditional_fluxes', 'avg_rates', 
'avg_state_probs', 'avg_total_fluxes', 'color_prob_evolution', 'conditional_arrivals',
'conditional_flux_evolution', 'conditional_fluxes', 'duration_count', 'durations', 
'rate_evolution', 'state_labels', 'state_pop_evolution', 'target_flux_evolution', 'total_fluxes']

  /target_flux_evolution [window,state]
    Total flux into a given macro state based on
    windows of iterations of varying width, as in /rate_evolution.
  /conditional_flux_evolution [window,state,state]
    State-to-state fluxes based on windows of
    varying width, as in /rate_evolution.

The structure of these datasets is as follows:
  iter_start
    (Integer) Iteration at which the averaging window begins (inclusive).
  iter_stop
    (Integer) Iteration at which the averaging window ends (exclusive).
  expected
    (Floating-point) Expected (mean) value of the observable as evaluated within
    this window, in units of inverse tau.
  ci_lbound
    (Floating-point) Lower bound of the confidence interval of the observable
    within this window, in units of inverse tau.
  ci_ubound
    (Floating-point) Upper bound of the confidence interval of the observable
    within this window, in units of inverse tau.
  stderr
    (Floating-point) The standard error of the mean of the observable
    within this window, in units of inverse tau.
  corr_len
    (Integer) Correlation length of the observable within this window, in units
    of tau.

TODO:
    * fix mfpt plots
    * multi direct.h5/assign.h5 input and error using Bayesian bootstrapping
    * account for assign.h5 populations implicitly using averaging info from 
      direct.h5. E.g. auto use cumulative or window averaged assign.h5 populations.
    * option to apply the RED scheme
    
    Other plots:
    4 panel plot of P_A, P_B, rate_AB, rate_BA, all as function of WE iteration
    Autocorrelation plot
    Event durations and distributions
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

from .bootstrap import get_CR_multi
#from wekap.bootstrap import get_CR_multi

import sys
import importlib

class Kinetics:
    """
    Plot the fluxes and rates from direct.h5 files.
    """

    def __init__(self, direct=None, assign=None, statepop="direct", tau=100e-12, state=1, 
                 label=None, flux_units="rates", ax=None, savefig=None, color=None, x_units="iterations",
                 cumulative_avg=True, linewidth=None, linestyle="-", postprocess_func=None,
                 red=False, *args, **kwargs):
        """
        Parameters
        ----------
        direct : str
            Name of output direct.h5 file from WESTPA w_direct or w_ipa.
        assign : str
            Default None (and will search for a file names `assign.h5` in the same dir).
            Otherwise can specify a specific assign.h5 file. Needed for labeled population data.
        tau : float
            The resampling interval of the WE simualtion.
            This should be in seconds, default 100ps = 100 * 10^-12 (s).
        state : int
            State for flux calculations (flux into `state`), 0 = A and 1 = B.
        label : str
            Data label.
        statepop : str
            'direct' for state_population_evolution from direct.h5 or
            'assign' for labeled_populations from assign.h5.
        flux_units : str
            Can be `rates` (default) or `mfpts`.
        ax : mpl axes object
        savefig : str
            Path to optionally save the figure.
        color : str
            Color of the line plot. Default None for mpl tab10 colors.
        x_units : str
            Default "iterations" for WE iterations, use "moltime" for molecular time and "agg" for agg sim time.
        cumulative_avg : bool
            Set to True (default) when kinetics were calculated with cumulative averaging.
            Only relevant with assign.h5 state populations.
        linewidth : float
        linestyle : str
        postprocess_func : func
            User function to import.
        red : bool
            Optionally use flux evolution data calculated using the Rate from Event Durations (RED) scheme.
            Set to True to use the `red_flux_evolution` dataset.
        ** args
        ** kwargs
        """
        # try to read in direct.h5 file
        self.direct = direct
        self.assign = assign
        try:
            self.direct_h5 = h5py.File(direct, "r")
        except FileNotFoundError as e:
            raise ValueError(e)

        if assign is None:
            # solution for getting assign.h5 color/labeled population
            # from direct.h5 path
            self._find_assign_h5()
        else:
             self.assign_h5 = h5py.File(assign, "r")

        self.tau = tau
        self.state = state
        self.label = label
        self.flux_units = flux_units
        self.statepop = statepop
        self.color = color

        self.cumulative_avg = cumulative_avg
        #print(f"CAVG: {self.cumulative_avg}")

        # set state pop instance attrs
        self.get_state_pop()

        # create new fig or plot onto existing
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax
            self.fig = plt.gcf()

        self.savefig = savefig

        self.x_units = x_units
        self.linewidth = linewidth
        self.linestyle = linestyle
        self.postprocess_func = postprocess_func
        self.red = red
        self.kwargs = kwargs

    def _find_assign_h5(self):
        """
        Look for assign.h5 file in same dir as direct.h5 input.
        
        Updates
        -------
        self.assign_h5
        """
        try:
            # temp solution for getting assign.h5 color/labeled population
            # TODO: obv doesn't work when file is not called 'direct.h5'
            self.assign_h5 = h5py.File(self.direct[:-9] + "assign.h5", "r")
        except (FileNotFoundError, TypeError) as e:
            print(f"{e}: Note that an assign.h5 file is needed only when using assign.h5 labeled_populations for the state populations.")

    def get_state_pop(self):
        """
        Update self.state_pops, self.state_pop_a, and self.state_pop_b based
        on self.direct_h5 and self.assign_h5
        """
        if self.statepop == "direct":
            # divide k_AB by P_A for equilibrium rate correction (AB and BA steady states)
            self.state_pops = np.array(self.direct_h5["state_pop_evolution"])
            # state A = label 0, state B = label 1
            self.state_pop_a = np.array([expected[2] for expected in self.state_pops[:,0]])
            self.state_pop_b = np.array([expected[2] for expected in self.state_pops[:,1]])
        elif self.statepop == "assign":
            # divide k_AB by P_A for equilibrium rate correction (AB and BA steady states)
            self.state_pops = np.array(self.assign_h5["labeled_populations"])

            # when using cumulative averaging and assign.h5 statepops
            if self.cumulative_avg is True:
                # Replace 0 with the index of your source state here, 
                # the order you defined states in west.cfg.
                state_pop = self.assign_h5['labeled_populations'][:,0]
                temp = np.sum(state_pop, axis=1)
                #state_pop_cum_avg = np.cumsum(temp) / np.arange(1, len(temp)+1)
                self.state_pop_a = np.cumsum(temp) / np.arange(1, len(temp)+1)

                # state b
                state_pop = self.assign_h5['labeled_populations'][:,0]
                temp = np.sum(state_pop, axis=1)
                #state_pop_cum_avg = np.cumsum(temp) / np.arange(1, len(temp)+1)
                self.state_pop_a = np.cumsum(temp) / np.arange(1, len(temp)+1)
            # or if you're using e.g. instantaneous fluxes, can grab the state pops directly
            else:
                # state A = label 0, state B = label 1
                #self.state_pop_a = np.array([expected[0] for expected in self.state_pops[:,0]])
                #self.state_pop_a = np.sum(self.state_pops[:,0], axis=1)
                #print(state_pop_a)
                #np.sum(self.state_pop_a)
                self.state_pop_a = np.sum(self.state_pops[:,0], axis=1)
                self.state_pop_b = np.sum(self.state_pops[:,1], axis=1)

    def extract_rate(self):
        """
        Get the raw rate array from one direct.h5 file.
        
        Returns
        -------
        rate_ab, ci_lb_ab, ci_ub_ab
        """
        # flux evolution dataset from cumulative evolution mode:
        # When calculating time evolution of rate estimates, 
        # ``cumulative`` evaluates rates over windows starting with --start-iter and 
        # getting progressively wider to --stop-iter by steps of --step-iter.
        fluxes = np.array(self.direct_h5["target_flux_evolution"])

        # conditional fluxes are macrostate to macrostate
        # 2 dimensions: [(0 -> 0, 0 -> 1), 
        #                (1 -> 0, 1 -> 1)] 
        # I want 0 -> 1
        #fluxes = np.array(h5["conditional_flux_evolution"])[:,:,1]

        # third column (expected) of the state (A(0) or B(1)) flux dataset (flux into state b = 1)
        if self.red:
            flux_ab = np.array(self.direct_h5["red_flux_evolution"])
        else:
            flux_ab = np.array([expected[2] for expected in fluxes[:,self.state]])
        # CIs in rate (s^-1) format (divided by tau)
        ci_lb_ab = np.array([expected[3] for expected in fluxes[:,self.state]]) * (1/self.tau)
        ci_ub_ab = np.array([expected[4] for expected in fluxes[:,self.state]]) * (1/self.tau)

        # TODO: update to be a state assignment attr
        # norm by state pop A if calculating A --> B
        # if self.state == 1:
        #     state_pop = self.state_pop_a
        # # norm by state pop B if calculating B --> A
        # elif self.state == 0:
        #     state_pop = 1 - self.state_pop_a
        # # TODO: temp fix
        # else:
        #     state_pop = self.state_pop_a

        # assign the state of target flux flow
        if self.state == 1:
            state_pop = self.state_pop_a
        elif self.state == 0:
            state_pop = self.state_pop_b
        else:
            print("Currently only support state 0 or state 1.")

        # 2 different approaches here, can norm by state_pop_a (sum of weights in a)
        # but since 2 state system, could also use 1 - state_pop_b since all not in b are in a
        flux_ab = flux_ab / state_pop
        #flux_ab = flux_ab / state_pop_a
        #flux_ab = flux_ab / (1 - state_pop_b)

        # convert from tau^-1 to seconds^-1
        rate_ab = flux_ab * (1/self.tau)
        
        return rate_ab, ci_lb_ab, ci_ub_ab

    def _get_x_data(self, n_iters):
        """        
        Parameters
        ----------
        n_iters : int
            Total iteration number to calc X-axis units from.
        
        Returns
        -------
        x_data : array
            Units will depend on self.x_units
        """
        # X-axis labels: WE iterations
        iterations = np.arange(0, n_iters, 1)
        if self.x_units == "iterations":
            x_data = iterations
        elif self.x_units == "moltime":
            # multiply by tau seconds converted to ps
            x_data = np.multiply(iterations, (self.tau * 1e12))
            # convert to ns
            x_data = np.divide(x_data, 1000)
        # agg sim time from assign.h5 data
        elif self.x_units == "agg":
            # tau per iter
            #npts = self.assign_h5["npts"][:]
            # segs per iter
            nsegs = self.assign_h5["nsegs"][:]
            # cumulative frames per iter
            x_data = np.cumsum(self.tau * nsegs)
            # convert from per second to per microsecond
            x_data *= 1e6
        else:
            raise ValueError(f"input x_units={self.x_units}, must be `iterations`, `moltime`, or `agg`.")
        return x_data

    def plot_rate(self):
        """
        Plot the rate constant = target flux evolution AB / P_A 

        Returns
        -------
        rate_ab : ndarray
            Array of rates from A -> B in seconds^-1.
        """
        rate_ab, ci_lb_ab, ci_ub_ab = self.extract_rate()

        # grab data for x-axis
        x_data = self._get_x_data(len(rate_ab))

        if self.flux_units == "mfpts":
            mfpt_ab = 1 / rate_ab
            self.ax.plot(x_data, mfpt_ab, label=self.label,
                         linewidth=self.linewidth, linestyle=self.linestyle)
            #ax.fill_between(x_data, mfpt_ab - (1/ci_lb_ab), mfpt_ab + (1/ci_ub_ab), alpha=0.5)
            #self.ax.set_ylabel("MFPT ($s$)")
        elif self.flux_units == "rates":
            self.ax.plot(x_data, rate_ab, color=self.color, label=self.label, 
                         linewidth=self.linewidth, linestyle=self.linestyle)
            self.ax.fill_between(x_data, rate_ab - ci_lb_ab, rate_ab + ci_ub_ab, alpha=0.25,
                                 color=self.color)
            #self.ax.set_ylabel("Rate Constant ($s^{-1}$)")
        else:
            raise ValueError(f"input flux_units={self.flux_units}, must be `rates` or `mfpts`.")

        self.format_rate_plot()

        return rate_ab

    def format_rate_plot(self):
        """
        General formatting options for rate plots.
        """
        if self.flux_units == "mfpts":
            self.ax.set_ylabel("MFPT ($s$)")
        elif self.flux_units == "rates":
            self.ax.set_ylabel("Rate Constant ($s^{-1}$)")

        if self.x_units == "iterations":
            self.ax.set_xlabel("WE Iteration")
        elif self.x_units == "moltime":
            self.ax.set_xlabel("Molecular Time ($ns$)")
        elif self.x_units == "agg":
            self.ax.set_xlabel("Aggregate Simulation Time ($\mu$$s$)")
        
        self.ax.set_yscale("log", subs=[2, 3, 4, 5, 6, 7, 8, 9])

    def plot_statepop(self):
        """
        Plot the state populations
        """
        # # divide k_AB by P_A for equilibrium rate correction (AB and BA steady states)
        # state_pops = np.array(self.direct_h5["state_pop_evolution"])
        # state A = label 0, state B = label 1
        #state_pop_a = np.array([expected[2] for expected in self.state_pops[:,0]])
        #state_pop_b = np.array([expected[2] for expected in self.state_pops[:,1]])
        # state_pop_a = np.sum(self.state_pops[:,0], axis=1)
        # state_pop_b = np.sum(self.state_pops[:,1], axis=1)

        # WE iterations
        iterations = np.arange(0, len(self.state_pop_a), 1)

        # plot both state population evolutions
        self.ax.plot(iterations, self.state_pop_a, label="State A")
        self.ax.plot(iterations, self.state_pop_b, label="State B")
        self.ax.set_xlabel(r"WE Iteration ($\tau$=100ps)")
        self.ax.set_ylabel("State Population")

        #return self.state_pop_a, self.state_pop_b

    def plot_exp_vals(self, ax=None, f_range=False, d2d1=False, f_range_all=False):
        """
        f_range : bool
            Set to True to use mark 25-67 s^-1 as the k_D1D2 rate.
        d2d1 : bool
            Set to True to also include k_D2D1.
        """
        if ax is None:
            ax = self.ax
        if self.flux_units == "rates":
            if f_range_all:
                # ax.axhline(60, alpha=1, color="tab:orange", label="4F k$_{D1D2}$", ls="--")
                # ax.axhline(25, alpha=1, color="tab:green", label="7F k$_{D1D2}$", ls="--")
                # ax.axhline(60, alpha=1, color="tab:red", ls="--")
                # ax.axhline(25, alpha=1, color="tab:green", ls="--")
                ax.axhline(60, alpha=1, color="tab:orange", ls="--")
                ax.axhline(25, alpha=1, color="tab:green", ls="--")
            elif f_range:
                # DTY 19F rates of 25-60 for k_D1D2
                ax.axhspan(25, 60, alpha=0.25, color="grey", label="NMR k$_{D1D2}$")
                if d2d1:
                    ax.axhspan(135, 179, alpha=0.25, color="tan", label="NMR k$_{D2D1}$")
            else:
                # D1-->D2 ~ 20-50, D2-->D1 ~ 100-150
                ax.axhline(60, color="k", ls="--", label="k$_{D2D1}$")
                if d2d1:
                    ax.axhline(134, color="red", ls="--", label="k$_{D1D2}$")
        elif self.flux_units == "mfpts":
            # converted to mfpt = 1 / rate
            ax.axhline(1/150, color="k", ls="--", label="MFPT$_{D2D1}$")
            if d2d1:
                ax.axhline(1/25, color="red", ls="--", label="MFPT$_{D1D2}$")
        else:
            raise ValueError(f"You put {self.flux_units} for flux_unit, which must be `mfpts` or `rates`.") 

    def _unpack_plot_options(self):
        """
        Unpack the plot_options kwarg dictionary.
        """
        # unpack plot options dictionary making sure not None
        # TODO: put all in ax.set()?
        #for key, item in self.plot_options.items():
        for key, item in self.kwargs.items():
            if key == "xlabel" and item:
                self.ax.set_xlabel(item)
            if key == "ylabel" and item:
                self.ax.set_ylabel(item)
            if key == "xlim" and item:
                self.ax.set_xlim(item)
            if key == "ylim"and item:
                self.ax.set_ylim(item)
            if key == "title" and item:
                self.ax.set_title(item)
            if key == "suptitle" and item:
                plt.suptitle(item)
            if key == "grid" and item:
                self.ax.grid(item, alpha=0.5)
            
            # now allowing for a list of line inputs
            if key == "axvline" and item:
                # make into list if not already
                if not isinstance(item, list):
                    item = [item]
                # loop each list item and plot line
                for line in item:
                    self.ax.axvline(line, color=self.color, linewidth=self.linewidth, linestyle=self.linestyle)
            if key == "axhline" and item:
                # make into list if not already
                if not isinstance(item, list):
                    item = [item]
                # loop each list item and plot line
                for line in item:
                    self.ax.axhline(line, color=self.color, linewidth=self.linewidth, linestyle=self.linestyle)

    def _run_postprocessing(self):
        """
        Run the user-specified postprocessing function.
        """
        # Parse the user-specifed string for the module and class/function name.
        module_name, attr_name = self.postprocess_func.split('.', 1) 
        # import the module ``module_name`` and make the function/class 
        # accessible as ``attr``.
        #attr = getattr(importlib.import_module(module_name), attr_name) 
        attr = getattr(self.load_module(module_name, '.'), attr_name)
        # Call ``attr``.
        attr()

    @staticmethod
    def load_module(module_name, path=None):
        """Load and return the given module, recursively loading containing packages as necessary."""
        if module_name in sys.modules:
            return sys.modules[module_name]

        if path is None:
            return importlib.import_module(module_name)

        spec_components = list(reversed(module_name.split('.')))
        qname_components = []
        mod_chain = []
        while spec_components:
            next_component = spec_components.pop(-1)
            qname_components.append(next_component)

            try:
                parent = mod_chain[-1]
                path = parent.__path__
            except IndexError:
                parent = None

            qname = '.'.join(qname_components)

            if qname in sys.modules:
                module = sys.modules[qname]
            else:
                spec = importlib.machinery.PathFinder().find_spec(qname, path)

                if spec is None:
                    raise ImportError(f'No module named {qname}')

                module = importlib.util.module_from_spec(spec)

                if spec.name not in sys.modules:
                    sys.modules[spec.name] = module

                spec.loader.exec_module(module)

                # Make the module appear in the parent module's namespace
                if parent:
                    setattr(parent, next_component, module)

            mod_chain.append(module)

        return module

    def plot_multi_rates(self, multi_direct, plotting=True):
        """
        Plot multiple direct.h5 flux evolution datasets.
        Use Bayesian bootstrapping for error estimates.

        Parameters
        ----------
        multi_direct : list
            List of paths to multiple `direct.h5` files.

        Returns
        -------
        multi_k, multi_k_avg, multi_k_uncertainty
        """
        # multi rate list per replicate
        multi_k = []

        # calc and append the 1st item which is the rates
        for dh5 in multi_direct:
            # update direct_h5, assign_h5, and state_pops before extracting new rate
            self.direct_h5 = h5py.File(dh5, "r")
            self.assign = None
            self._find_assign_h5()
            self.get_state_pop()
            multi_k.append(self.extract_rate()[0])

        # arithmetic mean
        multi_k_avg = np.average(multi_k, axis=0)
        # credibility regions from n_repeat trials Bayesian bootstrapping
        multi_k_uncertainty = get_CR_multi(multi_k, repeat=1000)

        # other error options (TODO: note these are avg+/-)
        #print(multi_k_uncertainty.shape)
        #multi_k_std = np.std(multi_k, axis=0)
        #multi_k_stderr = multi_k_std / np.sqrt(len(multi_k))
        #multi_k_stderr = np.rot90(np.vstack((multi_k_stderr,multi_k_stderr)))
        #print(multi_k_stderr.shape)

        # X-axis
        x_data = self._get_x_data(len(multi_k_avg))

        if plotting:
            # plot the replicates avg and error 
            self.ax.plot(x_data, multi_k_avg, color=self.color)
            self.ax.fill_between(x_data, multi_k_uncertainty[:,0],
                                multi_k_uncertainty[:,1], alpha=0.25, 
                                label=self.label, color=self.color)
            # self.ax.fill_between(iterations, multi_k_uncertainty[:,0],
            #                     multi_k_uncertainty[:,1], alpha=0.25, 
            #                     label=self.label, color='gray')

        # general formatting
        self.format_rate_plot()

        return x_data, multi_k, multi_k_avg, multi_k_uncertainty

if __name__ == "__main__":
#     #fig, ax = plt.subplots()
#     #k = Kinetics(f"D1D2_lt16oa/WT_v00/12oa/direct.h5", state=1, statepop="direct", ax=ax)
#     args = parse_arguments()

#     # plot style
#     if args.style == "default":
#         plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")
#     elif args.style is not None:
#         plt.style.use(args.style)

#     # make plot
#     k = Kinetics(**vars(args))
#     k.plot_rate()

#     # option to plot exp D1D2 values
#     if args.exp_values:
#         k.plot_exp_vals()
    
#     plt.tight_layout()
#     # option to save figure output
#     if args.savefig is not None:
#         plt.savefig(args.savefig)


    # testing multi direct.h5
    k = Kinetics("data/direct.h5")
    #k.plot_rate()
    k.plot_multi_rates(["data/direct.h5", "data/direct2.h5", "data/direct3.h5"])
    plt.show()

