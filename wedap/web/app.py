"""
Streamlit front-end for wedap / mdap / wekap.

A self-contained browser UI over the three analysis tools, each mirroring its
CLI (matplotlib style, axis/cbar labels, argument triaging) so plots look like
their command-line equivalents:

* **wedap** — WESTPA ``west.h5`` probability distributions + plots.
* **mdap**  — standard MD analysis data distributions + plots.
* **wekap** — WE kinetics: rates/MFPTs from a ``direct.h5`` file.

The tools live in three tabs. Each tab renders its own options (in a bordered
panel) and its plot below. Tabs are used rather than a sidebar selector so each
tool keeps its inputs when you switch away and back: Streamlit executes every
tab's body on every rerun (tab switching is client-side), so no widget's state
is dropped. A single shared sidebar could not hold three tools' controls at
once, and a sidebar radio that renders only the active tool would silently reset
the other tools' inputs whenever you switched.

Run locally:
    streamlit run wedap/web/app.py
or (with the web extra installed):
    wedap-web

Notes
-----
* Real ``west.h5`` files can be very large (GB-scale), so the default input mode
  is a **server-side path** rather than an upload.
* The expensive wedap pdist computation is cached with ``st.cache_data`` keyed on
  the file + pdist arguments, so display-only tweaks do not recompute the
  histogram. Tracing/gif/jointplot use the h5-backed path and bypass the cache.
"""
import matplotlib
matplotlib.use("Agg")  # headless rendering for the web server

import streamlit as st

from wedap.web._wedap_tab import render_wedap
from wedap.web._mdap_tab import render_mdap
from wedap.web._wekap_tab import render_wekap

TOOLS = {
    "wedap · WE H5 pdists": render_wedap,
    "mdap · MD data pdists": render_mdap,
    "wekap · WE kinetics": render_wekap,
}


def main():
    st.set_page_config(page_title="wedap", page_icon="📊", layout="wide")

    st.title("WEDAP — weighted ensemble & MD data analysis and plotting")
    st.caption("A browser-based front-end for the wedap, mdap, and wekap CLIs.")

    # Keep each tool's plot pinned while its (taller) options column is scrolled.
    # The plot content is wrapped in st.container(key="plot_sticky_*") (exposed as
    # a `st-key-plot_sticky_*` CSS class); we make the *column* that contains it
    # sticky. `align-self: flex-start` stops the flex row from stretching the
    # column to full height, which is what gives sticky room to actually pin.
    #
    # Also cap the content width so there's breathing room on the edges of wide
    # monitors. `max-width` (not a fixed width) means narrower viewports still use
    # all available space and reflow down — so it adapts as the window shrinks.
    st.markdown(
        """
        <style>
        div[data-testid="stMainBlockContainer"] {
            max-width: 1400px;
            margin-inline: auto;
            padding-left: 3rem;
            padding-right: 3rem;
        }
        @media (max-width: 900px) {
            div[data-testid="stMainBlockContainer"] {
                padding-left: 1rem;
                padding-right: 1rem;
            }
        }
        div[data-testid="stColumn"]:has([class*="st-key-plot_sticky"]) {
            position: sticky;
            top: 3rem;
            align-self: flex-start;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # one tab per tool; every tab body executes each rerun, so switching tabs
    # never drops a tool's option/input state
    for tab, render in zip(st.tabs(list(TOOLS)), TOOLS.values()):
        with tab:
            render()


if __name__ == "__main__":
    main()
