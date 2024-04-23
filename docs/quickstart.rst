Quickstart Guide
=============================

If you don't need the GUI, then you can just pip install.

``pip install wedap``

To install with the GUI:

``conda install -c conda-forge gooey``

``pip install wedap``

To do a developmental install, clone the github repo and run:

``git clone https://github.com/darianyang/wedap.git``

``cd wedap``

``pip install -e .``

in the cloned wedap directory.


Dependencies
**************
* numpy
* matplotlib
* h5py
* tqdm
* gif
* gooey (optional for GUI)

Usage Examples
**************
For more detailed examples, see the WEDAP Demo or Paper Figures Jupyter notebook tabs.

After installation, to run the CLI version and view available options:

``wedap --help``

Or:

``wedap -h``

To start the GUI simply input:

``wedap``

To start the GUI on MacOSX:

``pythonw /Path/to/wedap/git/repo/wedap/wedap``

To visualize the evolution of the pcoord for the example p53.h5 file via CLI:

``wedap -h5 wedap/data/p53.h5``

To do the same with the API:

.. code-block:: python

    import wedap
    import matplotlib.pyplot as plt

    wedap.H5_Plot(h5="wedap/data/p53.h5", data_type="evolution").plot()
    plt.show()


The resulting ``p53.h5`` file evolution plot will look like this:

.. raw:: html

    <p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/examples/p53_evo.png?raw=true" alt="p53 evo plot" width="400">
    </p>

See the examples directory for more realistic applications using the Python API.

Evolution plots are created by default using the CLI and GUI but average and instant probability distribution options are also available. To use one of your auxiliary datasets instead of the progress coordinate, just include the name of the aux dataset from your h5 file in the ``--Xname`` or ``--Yname`` fields:

``wedap -h5 wedap/data/p53.h5 --data_type average --Xname dihedral_10 --Yname dihedral_11``

Or:

``wedap -h5 wedap/data/p53.h5 -dt average -X dihedral_10 -Y dihedral_11``

The resulting ``p53.h5`` file average plot of the dihedral aux datasets will look like this:

.. raw:: html

    <p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/examples/p53_avg_aux.png?raw=true" alt="p53 avg aux plot" width="400">
    </p>

If you used a multi-dimensional progress coordinate and you want to use your pcoord for both the X and Y dimensions in a 2D average or instant plot, just use ``pcoord`` with the corresponding index set to the appropriate dimension (this also works with aux datasets which may have an additional dimension):

``wedap -h5 wedap/data/p53.h5 --data_type average --Xname pcoord --Xindex 0 --Yname pcoord --Yindex 1``

Or:

``wedap -h5 wedap/data/p53.h5 -dt average -X pcoord -Xi 0 -Y pcoord -Yi 1``

Or (since the default X options are the first pcoord, only the second pcoord needs to be specified):

``wedap -h5 wedap/data/p53.h5 -dt average -Y pcoord -Yi 1``

The resulting ``p53.h5`` file average plot of the pcoord datasets will look like this:

.. raw:: html

    <p align="left">
    <img src="https://github.com/darianyang/wedap/blob/main/examples/p53_avg_pcoord.png?raw=true" alt="p53 avg pcoord plot" width="400">
    </p>
