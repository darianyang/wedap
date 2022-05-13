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
- Pandas

### GUI

wedap has a GUI built using [Gooey](https://github.com/chriskiehl/Gooey) which can be launched by running `pythonw wedap.py` (on MacOSX) or `python wedap.py` with no arguments. If you wish to use the command line interface instead include `--ignore-gooey`

### Installation
First install the dependencies
``` bash
conda env create --name wedap --file requirements.txt
conda activate wedap
conda install -c conda-forge gooey
```
Or update an existing environmnent
``` bash
conda env update ENV_NAME --file requirements.txt
```

For now, instead of pip installing or using setup.py (these will be available later), you could try just setting an alias to `wedap/wedap.py`. This could be done with the following bash command from the main wedap directory containing this README.
``` bash
$ echo "alias wedap=\"python3 $PWD/wedap/wedap.py\"" >> ~/.bash_aliases 
$ source ~/.bashrc
```

### Examples

To run the CLI version and view available options:
``` Bash
python3 wedap.py --ignore-gooey --help
```
If you have the alias set up:
``` Bash
wedap --ignore-gooey --help
```

### Contributing

Features should be developed on branches. To create and switch to a branch, use the command:

`git checkout -b new_branch_name`

To switch to an existing branch, use:

`git checkout branch_name`

To submit your feature to be incorporated into the main branch, you should submit a `Pull Request`. The repository maintainers will review your pull request before accepting your changes.

### Copyright

Copyright (c) 2022, Darian Yang
