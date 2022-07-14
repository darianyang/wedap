wedap
===========================
![tests](https://github.com/darianyang/wedap/actions/workflows/tests.yml/badge.svg)
[![PyPI version](https://badge.fury.io/py/wedap.svg)](https://badge.fury.io/py/wedap)
[![Downloads](https://pepy.tech/badge/wedap)](https://pepy.tech/project/wedap)
[![GitHub license](https://img.shields.io/github/license/darianyang/wedap)](https://github.com/darianyang/wedap/blob/master/LICENSE)

Weighted Ensemble data analysis and plotting.

This is used to plot H5 files produced from running [WESTPA](https://github.com/westpa/westpa).

This repository is currently under development.

### Requirements

- Numpy
- Matplotlib
- H5py
- Moviepy
- Scipy
- Gooey
- tqdm

### GUI

wedap has a GUI built using [Gooey](https://github.com/chriskiehl/Gooey) which can be launched by running `wedap` or `python wedap` if you're in the main wedap directory of this repository. If you're using MacOSX, you'll need to run `pythonw wedap` in the main directory since conda prevents wxPython from accessing the display on Mac. If you pip install (instead of conda isntall) wxPython and Gooey on Mac you may be able to just run `wedap`. If you wish to use the command line interface instead include the `--ignore-gooey` flag.

### Installation
I recommend first installing dependencies via conda, especially gooey.
To install the dependencies into your python env via pip or conda:
``` bash
conda env create --name wedap --file requirements.txt
conda activate wedap
conda install -c conda-forge gooey
pip install wedap
```
Or update an existing environmnent:
``` bash
conda activate ENV_NAME
conda env update ENV_NAME --file requirements.txt
conda install -c conda-forge gooey
pip install wedap
```
Or pip install (you may have issues pip installing wxPython):
``` bash
pip install gooey
pip install wedap
```
If you have the repository cloned, go into the main wedap directory:
``` bash
conda install -c conda-forge gooey
pip install .
```

Note that gooey is kindof troublesome to pip install in some systems, which is why it's not included in the requirements (although it is required). I am trying to fix this but for now I reccomend conda installing gooey.

For MacOSX, you can set up an alias in your `.bash_profile` by running the following:
```
echo "alias wedap=pythonw /Path/to/wedap/git/repo/wedap/wedap" >> ~/.bash_profile
```
Then simply type `wedap` on the terminal to run the wedap GUI.

### Examples

After installation, to run the CLI version and view available options:
``` bash
wedap --ignore-gooey --help
```
To start the GUI simply input:
``` bash
wedap
```
To start the GUI on MacOSX:
``` bash
pythonw /Path/to/wedap/git/repo/wedap/wedap
```
To visualize the evolution of the pcoord for the example p53.h5 file via CLI:
``` bash
wedap --ignore-gooey -h5 /path/to/h5/p53.h5
```
To do the same with the API:
``` Python
import wedap
import matplotlib.pyplot as plt

wedap.H5_Plot(h5="/path/to/h5/p53.h5", data_type="evolution").plot()
plt.show()
```
See the examples directory for more realistic applications using the Python API.

### Contributing

Features should be developed on branches. To create and switch to a branch, use the command:

`git checkout -b new_branch_name`

To switch to an existing branch, use:

`git checkout branch_name`

To submit your feature to be incorporated into the main branch, you should submit a `Pull Request`. The repository maintainers will review your pull request before accepting your changes.

### Copyright

Copyright (c) 2022, Darian Yang
