### Scripts from https://github.com/ZuckermanLab/BayesianBootstrap

Kinetic measurements may result in a small set of high-variance data. Such data are not readily amenable to standard uncertainty analysis. The Bayesian bootstrap provides a credibility region to the true mean value that is not biased toward small values and is logically more consistent with the given type of data.

For more information please read and cite the corresponding publication: https://doi.org/10.1021/acs.jctc.9b00015

#### Example
The example directory contains 5 sample data files with arbitrary time values (column 1) and example rate constants that span many orders of magnitude (column 2). The python script performs a 10,000-fold Bayesian bootstrap of the 5 different rate constants at any time point and writes to an output file the average rate constant and the minimum and maximum values of a 95% credibility region.
