"""
Optional Streamlit web front-end for wedap.

Run with:
    streamlit run wedap/web/app.py
or, once installed with the ``web`` extra (``pip install wedap[web]``):
    wedap-web
"""
import os
import sys


def launch():
    """
    Console-script entry point (``wedap-web``) that boots the Streamlit server
    on this package's app.py. Equivalent to ``streamlit run .../web/app.py``.
    Any extra CLI args are forwarded to Streamlit.
    """
    try:
        from streamlit.web import cli as st_cli
    except ImportError:
        sys.exit(
            "Streamlit is not installed. Install the web extra with:\n"
            "    pip install 'wedap[web]'"
        )
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path, *sys.argv[1:]]
    sys.exit(st_cli.main())
