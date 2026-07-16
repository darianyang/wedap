"""
Shared logging utilities for wedap, mdap, and wekap.

Provides a single place to configure logging verbosity from the command line
(``--verbose``/``--debug``) and a helper to echo the resolved constructor call
so that a CLI invocation can be lifted directly into a Python script or Jupyter
notebook.

Example
-------
Running a command with ``--verbose`` will print the equivalent Python call, e.g.::

    $ wekap -dh5 direct.h5 --tau 1e-10 --verbose
    INFO | wekap | Kinetics(direct='direct.h5', tau=1e-10, state=1, ...)

which can be copy-pasted into a script as ``wekap.Kinetics(...)``.
"""

import logging

# concise formatter shared by all three packages
_FORMAT = "%(levelname)s | %(name)s | %(message)s"

# packages whose loggers we manage
_MANAGED = ("wedap", "mdap", "wekap")


def get_logger(name):
    """
    Return a logger for the given module/package name.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


def set_log_level(verbose=False, debug=False):
    """
    Configure the root logging level for the wedap/mdap/wekap loggers.

    Default (neither flag set) is WARNING so normal runs stay quiet.

    Parameters
    ----------
    verbose : bool
        If True, set level to INFO.
    debug : bool
        If True, set level to DEBUG (takes precedence over ``verbose``).
    """
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    # attach a single stderr handler to each managed package logger
    for pkg in _MANAGED:
        pkg_logger = logging.getLogger(pkg)
        pkg_logger.setLevel(level)
        # avoid duplicate handlers if called more than once
        if not any(getattr(h, "_wedap_handler", False) for h in pkg_logger.handlers):
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(_FORMAT))
            handler._wedap_handler = True
            pkg_logger.addHandler(handler)
        # don't double-print through the root logger
        pkg_logger.propagate = False


def format_call(cls_name, kwargs):
    """
    Format a class/function call string from a kwargs mapping so it can be
    copy-pasted into a Python script or notebook.

    Parameters
    ----------
    cls_name : str
        Name of the class or function, e.g. ``"Kinetics"`` or ``"H5_Plot"``.
    kwargs : dict
        Mapping of argument names to values. Keys with a value of None that are
        purely internal (e.g. argparse plumbing) can be pre-filtered by the caller.

    Returns
    -------
    str
        e.g. ``Kinetics(direct='direct.h5', tau=1e-10, state=1)``
    """
    parts = ", ".join(f"{key}={value!r}" for key, value in kwargs.items())
    return f"{cls_name}({parts})"
