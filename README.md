<p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/docs/_static/wedap_logo.png?raw=true" alt="wedap logo" width="400">
</p>

![tests](https://github.com/darianyang/wedap/actions/workflows/tests.yml/badge.svg)
[![docs](https://img.shields.io/website?label=docs&up_color=brightgreen&up_message=online&url=https%3A%2F%2Fdarianyang.github.io%2Fwedap%2Fdocs%2Fhtml%2Findex.html)](https://darianyang.github.io/wedap/docs/html/index.html)
[![PyPI version](https://badge.fury.io/py/wedap.svg)](https://badge.fury.io/py/wedap)
[![Downloads](https://static.pepy.tech/badge/wedap)](https://pepy.tech/project/wedap)
[![GitHub license](https://img.shields.io/github/license/darianyang/wedap)](https://github.com/darianyang/wedap/blob/master/LICENSE)

**WEDAP** : **w**eighted **e**nsemble **d**ata **a**nalysis and **p**lotting (pronounced we-dap)

`wedap` is primarily used to plot H5 files produced from running [WESTPA](https://github.com/westpa/westpa).

`mdap` can be used to plot data files from analysis of standard MD simulations.

For a demo and summary of features, see this [jupyter notebook](docs/notebook/wedap_demo.ipynb).

Or view the same demo notebook on the [documentation web page](https://darianyang.github.io/wedap/docs/html/notebook/wedap_demo.html).

### Requirements

- numpy
- matplotlib
- h5py
- scipy
- tqdm

### Optional

- gif (optional for making gifs)
- gooey (optional for GUI)

## Installation

If you don't need the GUI, then installing `Gooey` is not required and you can just pip install.
``` bash
pip install wedap
```
Otherwise you can install with `Gooey`, e.g. into a new conda env:
``` bash
conda env create --name wedap python=3.8+
conda activate wedap
conda install -c conda-forge gooey
pip install wedap
```
Or update an existing environmnent:
``` bash
conda activate ENV_NAME
conda install -c conda-forge gooey
pip install wedap
```

Note that `Gooey` is kindof troublesome to pip install in some systems, which is also why it's not included in the requirements (although it is required for the GUI). For now, I recommend conda installing `Gooey`.

## GUI

`wedap` has a GUI built using [Gooey](https://github.com/chriskiehl/Gooey) which can be launched from the command line by simply running 
``` bash
wedap
```
or 
`python wedap` if you're in the main `wedap` directory of this repository. 


If you're using MacOSX, you'll need to run `pythonw wedap` in the main directory since conda prevents wxPython from accessing the display on Mac. 
If you pip install (instead of conda installing) `wxPython` and `Gooey` on Mac you may be able to just run `wedap`. 

For MacOSX, you can set up an alias in your `.bash_profile` by running the following:
``` bash
echo "alias wedap=pythonw /Path/to/wedap/git/repo/wedap/wedap" >> ~/.bash_profile
```
Then simply type `wedap` in the terminal to run the wedap GUI.

## Examples

After installation, to run the CLI version and view available options:
``` bash
wedap --help
```
Or:
``` bash
wedap -h
```
To start the GUI simply input:
``` bash
wedap
```
To start the GUI on MacOSX:
``` bash
pythonw /"Path to wedap git repo"/wedap/wedap
```
To visualize the evolution of the pcoord for the example p53.h5 file via CLI:
``` bash
wedap -h5 wedap/data/p53.h5
```
To do the same with the API:
``` Python
import wedap
import matplotlib.pyplot as plt

wedap.H5_Plot(h5="wedap/data/p53.h5", data_type="evolution").plot()
plt.show()
```
The resulting `p53.h5` file evolution plot will look like this:
<p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/examples/p53_evo.png?raw=true" alt="p53 evo plot" width="400">
</p>

See the examples directory for more realistic applications using the Python API.

Evolution plots are created by default using the CLI and GUI but average and instant probability distribution options are also available. To use one of your auxiliary datasets instead of the progress coordinate, just include the name of the aux dataset from your h5 file in the `--Xname` or `--Yname` fields:
``` bash
wedap -h5 wedap/data/p53.h5 --data_type average --Xname dihedral_10 --Yname dihedral_11
```
Or:
``` bash
wedap -h5 wedap/data/p53.h5 -dt average -X dihedral_10 -Y dihedral_11
```

The resulting `p53.h5` file average plot of the dihedral aux datasets will look like this:
<p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/examples/p53_avg_aux.png?raw=true" alt="p53 avg aux plot" width="400">
</p>

If you used a multi-dimensional progress coordinate and you want to use your pcoord for both the X and Y dimensions in a 2D average or instant plot, just use `pcoord` with the corresponding index set to the appropriate dimension (this also works with aux datasets which may have an additional dimension):
``` bash
wedap -h5 wedap/data/p53.h5 --data_type average --Xname pcoord --Xindex 0 --Yname pcoord --Yindex 1
```
Or:
``` bash
wedap -h5 wedap/data/p53.h5 -dt average -X pcoord -Xi 0 -Y pcoord -Yi 1
```
Or (since the default X options are the first pcoord, only the second pcoord needs to be specified):
``` bash
wedap -h5 wedap/data/p53.h5 -dt average -Y pcoord -Yi 1
```

The resulting `p53.h5` file average plot of the pcoord datasets will look like this:
<p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/examples/p53_avg_pcoord.png?raw=true" alt="p53 avg pcoord plot" width="400">
</p>

## Motivation
`WESTPA` already comes with some excellent analysis tools for generating probability distributions, so why is `wedap` needed?

`wedap` was originally built as a way to simplify the original `WESTPA` plotting pipeline:

Native `WESTPA` CLI-based Analysis Tools:

    ┌───────┐       w_pdist        ┌────────┐        plothist         ┌────────┐
    │west.h5├─────────────────────►│pdist.h5├────────────────────────►│plot.pdf│
    └───────┘ --construct-dataset  └────────┘ --postprocess-function  └────────┘
                   module.py                      plot_settings.py


Analysis using `wedap`:

    ┌───────┐     wedap       ┌────────┐
    │west.h5├────────────────►│plot.pdf│
    └───────┘ CLI/GUI/Python  └────────┘

So `wedap` can generate plots with more flexibilty and less intermediate files, providing an especially useful way to plot aux datasets and explore your h5 file. 
* The Python interface allows for advanced users to quickly generate a plot as a matplotlib axes object which can be further customized all in one Python script.
    * For example, the `moviepy` or `gif` package can be used with wedap to easily create a gif of your h5 file (see an example of this in `wedap/h5_movie.py`).
    * The actual data can also be easily extracted and then analyzed (see `wedap/h5_cluster.py` for an example of k-means clustering using the data from a WESTPA west.h5 file). 
* The GUI allows for users who may not be comfortable with command line tools or Python to be able to quickly analyze their simulation results.
* A CLI is also available if using wedap on a system without access to a display.

Since the original implementation of `wedap`, many more features have been added that are not available using the `WESTPA` `w_pdist` and `plothist` tools, these include the following:
* Easy WE tracing and plotting by inputing an iteration and segment, or by inputing the X and Y value to then query and trace.
* 3D plots that replace the probability with another pcoord or aux dataset (`plot_mode="scatter3d"`).
* Selective basis states (if you have multiple basis states, only plot the probability contributions from specific states).
    * See the `skip_basis` argument (available through the Python API only currently).
* More to come!

Note that the `WESTPA` analysis tools have features not available in `wedap` and may still be of interest to you.

## Contributing

Have an idea for a feature to add to wedap? Let me know and I may be able to incorporate it (dty7@pitt.edu).

Or feel free to try developing it yourself! Features should be developed on branches. To create and switch to a branch, use the command:

`git checkout -b new_branch_name`

To switch to an existing branch, use:

`git checkout branch_name`

To submit your feature to be incorporated into the main branch, you should submit a `Pull Request`. The repository maintainers will review your pull request before accepting your changes.

## Copyright

Copyright (c) 2021, Darian Yang
