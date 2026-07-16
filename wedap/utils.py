"""
Small shared helpers for wedap, mdap, and wekap.
"""

from importlib import resources

import matplotlib.pyplot as plt


def apply_style(style, package):
    """
    Apply a matplotlib style for the given package's command-line entry point.

    Consolidates the style-loading logic previously duplicated across the three
    ``__main__.py`` files. Reads the packaged ``styles/default.mplstyle`` via
    ``importlib.resources`` (no temp file needed).

    Parameters
    ----------
    style : str or None
        "default" to use the package's bundled default.mplstyle, "None"/None for
        basic matplotlib settings, or any other value passed straight to
        ``plt.style.use`` (a named mpl style or a path to a .mplstyle file).
    package : str
        The package whose bundled style to use, e.g. "wedap", "mdap", or "wekap".
    """
    if style == "default":
        # locate the packaged default style file and apply it
        style_path = resources.files(package) / "styles" / "default.mplstyle"
        with resources.as_file(style_path) as path:
            plt.style.use(str(path))
    elif style not in (None, "None"):
        plt.style.use(style)
