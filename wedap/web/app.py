"""
Minimal Streamlit front-end for wedap.

This is a self-contained prototype exposing the most common wedap pdist +
plotting options through a browser UI. It intentionally covers only the core
`wedap` (WESTPA H5) workflow; `mdap`/`wekap` tabs can be added the same way.

Run locally:
    streamlit run wedap/web/app.py

Notes
-----
* Because real ``west.h5`` files can be very large (GB-scale), the default
  input mode is a **server-side path** rather than an upload. An upload option
  is provided for small files (e.g. the packaged ``p53.h5``).
* The expensive pdist computation is cached with ``st.cache_data`` keyed on the
  file + pdist arguments, so tweaking display-only options (cmap, limits) does
  not recompute the histogram.
"""
import io
import os
import tempfile

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless rendering for the web server
import matplotlib.pyplot as plt

import streamlit as st

import wedap


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def list_datasets(h5_path, first_iter=1):
    """
    Return (pcoord_options, aux_options) available in a west.h5 file.

    pcoord is always present; aux datasets are read from the first iteration's
    auxdata group if it exists.
    """
    aux = []
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(f"iterations/iter_{first_iter:08d}/auxdata")
            if grp is not None:
                aux = sorted(grp.keys())
    except (OSError, KeyError):
        pass
    return ["pcoord"], aux


@st.cache_data(show_spinner="Computing probability distribution...")
def compute_pdist(h5_path, pdist_kwargs):
    """
    Run H5_Pdist and return (X, Y, Z). Cached on the file path + args so that
    display-only changes don't trigger a recompute.

    pdist_kwargs is passed as a plain dict (hashable-friendly) of constructor
    arguments for wedap.H5_Pdist.
    """
    X, Y, Z = wedap.H5_Pdist(h5=h5_path, **pdist_kwargs).pdist()
    return X, Y, Z


def render_plot(X, Y, Z, plot_kwargs):
    """Build the figure with H5_Plot using precomputed arrays and return it."""
    plot = wedap.H5_Plot(X=X, Y=Y, Z=Z, **plot_kwargs)
    plot.plot()
    return plot.fig


# --------------------------------------------------------------------------- #
# app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="wedap", page_icon="📊", layout="wide")
    st.title("wedap — weighted ensemble data analysis & plotting")
    st.caption(
        "Interactive front-end for plotting WESTPA H5 files. "
        "A browser-based alternative to the CLI."
    )

    # ---- data source ------------------------------------------------------ #
    with st.sidebar:
        st.header("Data source")
        source = st.radio(
            "Input mode",
            ["Server path", "Upload file", "Example (p53.h5)"],
            help="Large west.h5 files should use a server-side path; uploads are "
                 "held in memory and best for small files.",
        )

        h5_path = None
        if source == "Server path":
            h5_path = st.text_input("Path to west.h5 file", value="")
            if h5_path and not os.path.isfile(h5_path):
                st.warning("File not found at that path.")
                h5_path = None
        elif source == "Upload file":
            upload = st.file_uploader("west.h5 file", type=["h5", "hdf5"])
            if upload is not None:
                # persist to a temp file so h5py can open it by path
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
                tmp.write(upload.getbuffer())
                tmp.flush()
                h5_path = tmp.name
        else:  # packaged example
            example = os.path.join(os.path.dirname(wedap.__file__), "data", "p53.h5")
            h5_path = example if os.path.isfile(example) else None
            if h5_path is None:
                st.warning("Example p53.h5 not found in the installed package.")

    if not h5_path:
        st.info("Select a data source in the sidebar to begin.")
        st.stop()

    # ---- pdist options ---------------------------------------------------- #
    pcoords, aux = list_datasets(h5_path)
    dataset_options = pcoords + aux

    with st.sidebar:
        st.header("Pdist options")
        data_type = st.selectbox("data_type", ["evolution", "average", "instant"], index=0)
        p_units = st.selectbox(
            "p_units",
            ["kT", "kcal", "raw", "raw_norm", "raw_norm_tot"],
            index=0,
        )

        st.subheader("Axes")
        Xname = st.selectbox("Xname", dataset_options, index=0)
        Xindex = st.number_input("Xindex", min_value=0, value=0, step=1)
        # for evolution only X is used; Y optional otherwise
        use_y = data_type != "evolution"
        Yname = None
        Yindex = 0
        if use_y:
            y_choices = ["(none)"] + dataset_options
            y_sel = st.selectbox("Yname", y_choices, index=0)
            Yname = None if y_sel == "(none)" else y_sel
            Yindex = st.number_input("Yindex", min_value=0, value=0, step=1)

        st.subheader("Iterations")
        first_iter = st.number_input("first_iter", min_value=1, value=1, step=1)
        last_iter = st.number_input("last_iter (0 = all)", min_value=0, value=0, step=1)
        bins = st.slider("bins", min_value=20, max_value=300, value=100, step=10)

        st.header("Plot options")
        plot_mode = st.selectbox(
            "plot_mode",
            ["hist", "contourf", "contourl", "line", "scatter3d", "hexbin3d"],
            index=0,
        )
        cmap = st.text_input("cmap", value="viridis")
        alpha = st.slider("alpha", min_value=0.0, max_value=1.0, value=1.0, step=0.05)

    # assemble kwargs (dicts must be hashable-stable for the cache key)
    pdist_kwargs = dict(
        data_type=data_type,
        Xname=Xname,
        Xindex=int(Xindex),
        Yname=Yname,
        Yindex=int(Yindex),
        first_iter=int(first_iter),
        last_iter=None if last_iter == 0 else int(last_iter),
        bins=(int(bins), int(bins)),
        p_units=p_units,
        no_pbar=True,
    )

    plot_kwargs = dict(
        plot_mode=plot_mode,
        cmap=cmap or None,
        p_units=p_units,
    )
    if alpha < 1.0:
        plot_kwargs["alpha"] = alpha

    # ---- compute + render ------------------------------------------------- #
    try:
        X, Y, Z = compute_pdist(h5_path, pdist_kwargs)
    except Exception as exc:  # surface errors in the UI instead of crashing
        st.error(f"Pdist computation failed: {exc}")
        st.stop()

    try:
        fig = render_plot(X, Y, Z, plot_kwargs)
    except Exception as exc:
        st.error(f"Plotting failed: {exc}")
        st.stop()

    col_plot, col_info = st.columns([3, 1])
    with col_plot:
        st.pyplot(fig)
    with col_info:
        st.subheader("Equivalent Python")
        st.code(
            "import wedap\n"
            f"X, Y, Z = wedap.Pdist(\n"
            f"    h5={h5_path!r},\n"
            + "".join(f"    {k}={v!r},\n" for k, v in pdist_kwargs.items())
            + ").pdist()\n"
            f"wedap.Plot(X, Y, Z,\n"
            + "".join(f"    {k}={v!r},\n" for k, v in plot_kwargs.items())
            + ").plot()",
            language="python",
        )

        # download the current figure as a PNG
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button(
            "Download PNG",
            data=buf.getvalue(),
            file_name="wedap_plot.png",
            mime="image/png",
        )

    plt.close(fig)


if __name__ == "__main__":
    main()
