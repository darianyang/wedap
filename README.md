wedap
===========================
![tests](https://github.com/darianyang/fluorelax/actions/workflows/test.yml/badge.svg)

Weighted Ensemble data analysis and plotting.

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

wedap has a GUI built using [Gooey](https://github.com/chriskiehl/Gooey) which can be launched by running `pythonw wedap.py` (on MacOSX) or `python wedap.py` with no arguments. If you wish to use the command line interface instead include the `--ignore-gooey` flag.

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
pip install gooey
pip install .
```

Note that gooey is kindof troublesome to pip install in some systems, which is why it's not included in the requirements (although it is required). I am trying to fix this but for now I reccomend conda installing gooey.

### Examples

After installation, to run the CLI version and view available options:
``` bash
wedap --ignore-gooey --help
```
To start the GUI simply input:
``` bash
wedap
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
### Contributing

Features should be developed on branches. To create and switch to a branch, use the command:

`git checkout -b new_branch_name`

To switch to an existing branch, use:

`git checkout branch_name`

To submit your feature to be incorporated into the main branch, you should submit a `Pull Request`. The repository maintainers will review your pull request before accepting your changes.

### Copyright

Copyright (c) 2022, Darian Yang
