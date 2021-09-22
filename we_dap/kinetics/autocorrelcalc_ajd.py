#!/usr/bin/env python
# Written May 2016, Alex DeGrave
import h5py
import numpy
import matplotlib.pyplot as pyplot
import scipy.stats

class CorrelationAnalysis:
    def __init__(self, kineth5pathlist, tstate, fi=None, li=None):
        '''
        Calculate autocorrelation of weight/number flux into a target state.
        Currently, this script is only suitable for steady-state simulations.
        It considers only total flux/total "arrivals" (counts) into a target 
        state, rather than conditional flux/arrivals.  Additionally, this tool
        calculates error bars (95% CI) based on multiple independent WE
        simulations (rather than using bootstrapping).
        Output:
          (1) First iteration at which the mean +/- the 95% CI for the 
              autocorrelation of the number/weight flux crosses zero.
          (2) Plots the autocorrelation as a function of lag time on a given
              axis. 
        Arguments:
        
        kineth5pathlist:
          A list of paths (strings) to output files from w_kinetics. Supply 
          one path for each independent WE simulation.
        tstate:
          Monitor flux into target state index ``tstate``
        fi:
          The first iteration to include in the analysis (inclusive)
 
        li:
          The last iteration to include in the analysis (inclusive)
        '''
        self.kineth5pathlist = kineth5pathlist
        self.tstate = tstate
        self.fi = fi
        self.li = li
        self.open_kineth5_files()
    
    def open_kineth5_files(self):
        self.kineth5list = []
        for kineth5path in self.kineth5pathlist:
            self.kineth5list.append(h5py.File(kineth5path,'r+'))
        return

    def calculate_from_arrivals(self):
        '''
        Calculate the autocorrelation of the number flux into the target state.
 
        Returns:
          (1) autocorrel_mean: A numpy.ndarray where autocorrel_mean[i] gives
              the mean autocorrelation at a lag time of i iterations.
          (2) ci: A numpy.ndarray such that autocorrel_mean[i] +(-) ci[i] gives  
              the upper (lower) bounds for the 95% confidence interval.
        '''
        autocorrels = []
        for kineth5 in self.kineth5list:
            if self.fi is not None and self.li is not None:
                number_flux = kineth5['arrivals']\
                                     [self.fi-1:self.li, self.tstate]
            elif self.fi is not None:
                number_flux = kineth5['arrivals'][self.fi-1:, self.tstate]
            elif self.li is not None:
                number_flux = kineth5['arrivals'][:self.li, self.tstate]

            autocorrel = self.estimate_autocorrelation(number_flux) 
            autocorrels.append(autocorrel)
        autocorrel_arr = numpy.array(autocorrels)
        autocorrel_mean = autocorrel_arr.mean(axis=0)
        autocorrel_std = autocorrel_arr.std(axis=0)
        autocorrel_se  = autocorrel_std/numpy.sqrt(autocorrel_arr.shape[0])
        student_t = scipy.stats.t.interval(0.95, autocorrel_arr.shape[0]-1)
        print(student_t[1])
        ci = autocorrel_se*student_t[1]
        return autocorrel_mean, ci

    def calculate_from_flux(self):
        '''
        Calculate the autocorrelation of the weight flux into the target state.
 
        Returns:
          (1) autocorrel_mean: A numpy.ndarray where autocorrel_mean[i] gives
              the mean autocorrelation at a lag time of i iterations.
          (2) ci: A numpy.ndarray such that autocorrel_mean[i] +(-) ci[i] gives  
              the upper (lower) bounds for the 95% confidence interval.
        '''
        autocorrels = []
        for kineth5 in self.kineth5list:
            if self.fi is not None and self.li is not None:
                number_flux = kineth5['total_fluxes']\
                                     [self.fi-1:self.li, self.tstate]
            elif self.fi is not None:
                number_flux = kineth5['total_fluxes'][self.fi-1:, self.tstate]
            elif self.li is not None:
                number_flux = kineth5['total_fluxes'][:self.li, self.tstate]

            autocorrel = self.estimate_autocorrelation(number_flux) 
            autocorrels.append(autocorrel)
        autocorrel_arr = numpy.array(autocorrels)
        autocorrel_mean = autocorrel_arr.mean(axis=0)
        autocorrel_std = autocorrel_arr.std(axis=0)
        autocorrel_se  = autocorrel_std/numpy.sqrt(autocorrel_arr.shape[0])
        student_t = scipy.stats.t.interval(0.95, autocorrel_arr.shape[0]-1)
        ci = autocorrel_se*student_t[1]
        return autocorrel_mean, ci

    def estimate_autocorrelation(self, x):
        """
        http://stackoverflow.com/q/14297012/190597
        http://en.wikipedia.org/wiki/Autocorrelation#Estimation
        """
        n = len(x)
        variance = x.var()
        x = x-x.mean()
        r = numpy.correlate(x, x, mode = 'full')[-n:]
        assert numpy.allclose(r, numpy.array([(x[:n-k]*x[-(n-k):]).sum() \
                                              for k in range(n)]))
        result = r/(variance*(numpy.arange(n, 0, -1)))
        return result
        
    def plot(self, xlims=None, figname='autocorrel.pdf', ax=None, format_axis=True):
        '''
        Calculate and plot the autocorrelation of the weight and number flux.
   
        Keyword Arguments:
          xlims: a tuple denoting the x-limits for the fig (in units of tau)
 
          figname: a file path at which to save the figure.
          ax: plot the figure on the axis ``ax``
        '''
        fmean, fci = self.calculate_from_flux()
        amean, aci = self.calculate_from_arrivals()

        xs = numpy.arange(0, fmean.shape[0])
        if ax is None:
            fig, ax = pyplot.subplots(figsize=(7.25,4))
        else:
            fig = pyplot.gcf()

        ax.plot(xs, fmean, color='black', label='autocorrelation of weight flux')
        ax.fill_between(xs, fmean-fci, fmean+fci, color=(0,0,0,0.3), 
                        linewidth=0.0)

        ax.plot(xs, amean, color='blue', label='autocorrelation of number flux')
        ax.fill_between(xs, amean-aci, amean+aci, color=(0,0,1,0.3), 
                        linewidth=0.0)

        ax.set_ylim(-1,1)
        ax.axhline(y=0, color='black')
        if xlims is not None:
            ax.set_xlim(xlims)

        fcorrel_time = 0
        for xval in range(0,fmean.shape[0]):
            if fmean[xval]-fci[xval] <=0:
                fcorrel_time = xval
                break
        print("Correlation time based on weight flux is {:d} iterations".format(fcorrel_time))
        ax.axvline(x=fcorrel_time, color='gray', ls='--')
        ax.text(fcorrel_time+1, -0.8, '{:d}'.format(fcorrel_time), color='black')

        acorrel_time = 0
        for xval in range(0,amean.shape[0]):
            if amean[xval]-aci[xval] <=0:
                acorrel_time = xval
                break
        print("Correlation time based on number flux is {:d} iterations".format(acorrel_time))
        ax.axvline(x=acorrel_time, color='blue', ls='--')
        ax.text(acorrel_time+1, -0.8, '{:d}'.format(acorrel_time), color='blue')

        if format_axis:
            ax.set_xlabel(u'lag time (\u03C4)')
            ax.set_ylabel('autocorrelation of weight/number flux')
            ax.legend(frameon=False)
       
        return
        

def main():
    s7160N2NPpaths = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_1_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_2_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_3_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_4_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_5_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_6_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_7_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_8_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_9_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_N2NP_10_kinetics.h5']

    s7160NP2Npaths = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_1_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_2_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_3_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_4_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_5_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_6_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_7_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_8_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_9_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_71_60/analysis/kinetics_files/71_60_NP2N_10_kinetics.h5']

    s6071N2NPpaths = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_1_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_2_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_3_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_4_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_5_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_6_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_7_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_8_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_9_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_N2NP_10_kinetics.h5']

    s6071NP2Npaths = ['/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_1_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_2_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_3_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_4_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_5_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_6_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_7_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_8_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_9_kinetics.h5',
                     '/mnt/NAS2/SS_SWITCH/RADIAL_SIMS/switch_60_71/analysis/kinetics_files/60_71_NP2N_10_kinetics.h5']


   
    fig, (ax1, ax2, ax3, ax4) = pyplot.subplots(4,1, figsize=(7.25, 8))

    ca = CorrelationAnalysis(s7160N2NPpaths, 1, fi=1, li=2000)
    ca.plot(xlims=(0,400), figname='autocorrel_comparison.pdf', ax=ax1,
            format_axis=False)

    ca = CorrelationAnalysis(s7160NP2Npaths, 0, fi=1, li=2000)
    ca.plot(xlims=(0,400), figname='autocorrel_comparison.pdf', ax=ax2,
            format_axis=False)

    ca = CorrelationAnalysis(s6071NP2Npaths, 0, fi=1, li=2000)
    ca.plot(xlims=(0,400), figname='autocorrel_comparison.pdf', ax=ax3,
            format_axis=False)

    ca = CorrelationAnalysis(s6071N2NPpaths, 1, fi=1, li=2000)
    ca.plot(xlims=(0,400), figname='autocorrel_comparison.pdf', ax=ax4,
            format_axis=False)


    fig.subplots_adjust(bottom=0.2, hspace=0.0)

    ax1.legend(frameon=False)
    ax4.set_xlabel('lag time')
    ax4.text(0.05, 0.6, 'autocorrelation of weight/number flux into target state', 
             transform=fig.transFigure, ha='center', va='center', rotation='90')
    for ax in (ax1, ax2, ax3):
        ax.set_xticklabels(['' for i in range(4)])
    for ax in (ax2, ax3, ax4):
        ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', ''])
    ax1.text(1.05, 0.5, u"holo N':apo N\nN\u2192N'", transform=ax1.transAxes, 
             rotation='90', ha='center', va='center')
    ax2.text(1.05, 0.5, u"holo N':apo N\nN'\u2192N", transform=ax2.transAxes, 
             rotation='90', ha='center', va='center')
    ax3.text(1.05, 0.5, u"apo N':holo N\nN'\u2192N", transform=ax3.transAxes, 
             rotation='90', ha='center', va='center')
    ax4.text(1.05, 0.5, u"apo N':holo N\nN\u2192N'", transform=ax4.transAxes, 
             rotation='90', ha='center', va='center')

    pyplot.savefig('autocorrel_comparison.pdf')

if __name__ == '__main__':
    main()
