"""
mdap tab: probability distributions + plots from standard MD analysis data.

``mdap`` reads pre-calculated MD datasets (``.dat``/``.npy``/``.npz``/``.pkl``
text or binary files, typically ``frame value`` columns) rather than a WESTPA
H5 file, but plots them through the same ``H5_Plot`` machinery. This tab mirrors
the ``mdap`` CLI: ``pdist`` vs ``time`` data types, the 1D/2D/3D plot modes, and
the shared formatting options.
"""
import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

import mdap
from ._common import (
    MODE_DIM, PLOT_MODES, P_UNITS, P_UNITS_LABELS,
    parse_lim, apply_style_safe, run_postprocess, section,
    plot_formatting_controls, postprocess_controls, figure_download_button,
)

DATA_TYPES = ["pdist", "time"]


def _stem(path):
    """Filename without directory or extension, for default axis labels."""
    if not path:
        return ""
    return os.path.splitext(os.path.basename(path))[0]


def _save_upload(upload, suffix):
    """Persist an uploaded data file to a temp path and return it."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    return tmp.name


def validate_md(data_type, has_y, has_z, plot_mode):
    """Return an error string for an incompatible mdap selection, else None."""
    dim = MODE_DIM[plot_mode]
    if data_type == "time":
        if plot_mode != "line":
            return ("Time (timeseries) plots are 1D over frames; use plot mode 'line'.")
        return None
    # pdist
    if dim == 3:
        if not (has_y and has_z):
            return ("3D plot modes ('scatter3d'/'hexbin3d') need three datasets: set "
                    "an X, Y, and Z data file.")
        return None
    if dim == 2 and not has_y:
        return ("This is a 1D distribution (only an X dataset). Use plot mode 'line' or "
                "'bar', or add a Y data file for a 2D distribution.")
    if dim == 1 and has_y:
        return ("You provided a Y dataset (2D distribution) but a 1D plot mode. Use "
                "'hist'/contour, or clear the Y data file for a 1D 'line'/'bar' plot.")
    return None


def render_mdap():
    st.subheader("mdap — molecular dynamics data distributions & plots")
    st.caption("Plot pre-calculated MD analysis datasets (`.dat`/`.npy`/...) "
               "through the same plotting engine as wedap.")

    example_dir = os.path.join(os.path.dirname(mdap.__file__), "data")

    # `loaded` drives the label; `_has` (from the current widget selection) drives
    # the expanded state so the panel doesn't snap shut mid-edit when switching
    loaded = st.session_state.get("md_loaded")
    _src = st.session_state.get("md_source", "Server path(s)")
    _has = (_src == "Example")
    _has = _has or (_src == "Server path(s)" and os.path.isfile(st.session_state.get("md_xf", "") or ""))
    _has = _has or (_src == "Upload" and st.session_state.get("md_ux") is not None)
    with st.expander(f"Data source — {loaded}" if loaded else "Data source",
                     expanded=not _has):
        source = st.radio("Input mode", ["Server path(s)", "Upload", "Example"],
                          key="md_source",
                          help="Point at data files on the server, upload them, or use "
                               "the bundled example datasets.")

        Xname = Yname = Zname = None
        if source == "Server path(s)":
            Xname = st.text_input("X data file", value="", key="md_xf") or None
            Yname = st.text_input("Y data file (optional)", value="", key="md_yf") or None
            Zname = st.text_input("Z data file (optional)", value="", key="md_zf") or None
            for label, p in [("X", Xname), ("Y", Yname), ("Z", Zname)]:
                if p and not os.path.isfile(p):
                    st.warning(f"{label} file not found at that path.")
        elif source == "Upload":
            ux = st.file_uploader("X data file", key="md_ux")
            uy = st.file_uploader("Y data file (optional)", key="md_uy")
            uz = st.file_uploader("Z data file (optional)", key="md_uz")
            Xname = _save_upload(ux, "_" + ux.name) if ux else None
            Yname = _save_upload(uy, "_" + uy.name) if uy else None
            Zname = _save_upload(uz, "_" + uz.name) if uz else None
        else:  # example
            ex_x = os.path.join(example_dir, "rms_bb_nmr.dat")
            ex_y = os.path.join(example_dir, "rms_bb_xtal.dat")
            st.caption("Example: `rms_bb_nmr.dat` (X) and `rms_bb_xtal.dat` (Y).")
            use_2d = st.checkbox("2D (include Y)", value=True, key="md_ex2d")
            Xname = ex_x if os.path.isfile(ex_x) else None
            Yname = ex_y if (use_2d and os.path.isfile(ex_y)) else None

    # remember the loaded X file so the panel above shows it and collapses next run
    st.session_state["md_loaded"] = os.path.basename(Xname) if Xname else None

    if not Xname:
        st.info("Provide at least an X data file above to begin.")
        return

    opts_col, plot_col = st.columns([1, 1])
    with opts_col, st.container(border=True):
        section("Pdist options")
        c1, c2 = st.columns(2)
        data_type = c1.selectbox("data_type", DATA_TYPES, index=0, key="md_dt")
        p_units = c2.selectbox("p_units", P_UNITS, index=0, key="md_pu")

        section("Plot mode")
        # time is a timeseries -> force line so the UI stays coherent
        if data_type == "time":
            plot_mode = "line"
            st.caption("plot_mode: line (timeseries)")
        else:
            plot_mode = st.selectbox("plot_mode", PLOT_MODES, index=0, key="md_pm")
        dim = MODE_DIM[plot_mode]

        # Y/Z only meaningful for pdist; drop them for a timeseries
        if data_type == "time":
            Yname = Zname = None
        # if the user didn't supply Z but picked a 3D mode, Z stays None (validated below)

        section("Data indices & intervals")
        c1, c2, c3 = st.columns(3)
        Xindex = c1.number_input("Xindex", min_value=0, value=1, step=1, key="md_xi")
        Yindex = c2.number_input("Yindex", min_value=0, value=1, step=1, key="md_yi")
        Zindex = c3.number_input("Zindex", min_value=0, value=1, step=1, key="md_zi")
        c1, c2, c3 = st.columns(3)
        Xinterval = c1.number_input("Xint", min_value=1, value=1, step=1, key="md_xint")
        Yinterval = c2.number_input("Yint", min_value=1, value=1, step=1, key="md_yint")
        Zinterval = c3.number_input("Zint", min_value=1, value=1, step=1, key="md_zint")

        section("Frames & bins")
        c1, c2 = st.columns(2)
        first_frame = c1.number_input("first_frame (0=start)", min_value=0, value=0, step=1, key="md_ff")
        last_frame = c2.number_input("last_frame (0=end)", min_value=0, value=0, step=1, key="md_lf")
        c1, c2 = st.columns(2)
        bins = c1.number_input("bins", min_value=10, max_value=1000, value=100, step=10, key="md_bins")
        timescale = c2.number_input("timescale", min_value=1.0, value=1e6, step=1e5,
                                    format="%.0f", key="md_ts",
                                    help="Frames-to-time divisor for 'time' plots (ps→µs = 1e6).")

        # ---- shared formatting -------------------------------------------- #
        if data_type == "time":
            def_x, def_y = "Time", _stem(Xname)
        elif dim == 1:
            def_x, def_y = _stem(Xname), P_UNITS_LABELS.get(p_units, "P(x)")
        else:
            def_x, def_y = _stem(Xname), _stem(Yname)
        default_cbar = _stem(Zname) if (dim == 3 and Zname) else ""
        plot_kwargs, meta = plot_formatting_controls(
            "md", dim, plot_mode,
            default_xlabel=def_x, default_ylabel=def_y, default_cbar=default_cbar,
            show_weighted=False,  # MD pdists are unweighted
        )
        style = meta["style"]

        with st.expander("Advanced pdist", expanded=False):
            histrange_x = parse_lim(st.text_input("histrange_x (lo,hi)", value="", key="md_hrx"))
            histrange_y = parse_lim(st.text_input("histrange_y (lo,hi)", value="", key="md_hry"))

        pp_mode, pp_path, pp_code, pp_func = postprocess_controls("md")

    # ---- validate (message shown in the plot column below) ---------------- #
    err = validate_md(data_type, Yname is not None, Zname is not None, plot_mode)

    # ---- assemble kwargs -------------------------------------------------- #
    pdist_kwargs = dict(
        data_type=data_type,
        Xname=Xname, Xindex=int(Xindex), Xinterval=int(Xinterval),
        Yname=Yname, Yindex=int(Yindex), Yinterval=int(Yinterval),
        Zname=Zname, Zindex=int(Zindex), Zinterval=int(Zinterval),
        first_frame=None if first_frame == 0 else int(first_frame),
        last_frame=None if last_frame == 0 else int(last_frame),
        bins=(int(bins), int(bins)),
        p_units=p_units,
        T=int(meta["T"]),
        timescale=float(timescale),
        no_pbar=True,
    )
    if histrange_x is not None:
        pdist_kwargs["histrange_x"] = histrange_x
    if histrange_y is not None:
        pdist_kwargs["histrange_y"] = histrange_y

    plot_kwargs["p_units"] = p_units

    # MD analysis files are small, so recompute per render (like the CLI); MD_Plot
    # builds the pdist internally and also handles jointplot's raw re-normalization
    all_kwargs = {**pdist_kwargs, **plot_kwargs}

    apply_style_safe(style, "mdap")

    # ---- render in the sticky plot column: warnings/loading/plot together -- #
    with plot_col, st.container(key="plot_sticky_md"):
        notice = st.container()  # pinned above the plot with it
        if err:
            notice.warning(f"⚠️ {err}")
            return

        try:
            with st.spinner("Building distribution..."):
                plot = mdap.MD_Plot(**all_kwargs)
                plot.plot()
            if pp_mode != "none":
                try:
                    run_postprocess(plot, "file" if pp_mode == "file path" else "inline",
                                    pp_path, pp_code, pp_func, extra_ns={"mdap": mdap})
                except Exception as exc:
                    notice.warning(f"Postprocessing failed: {exc}")
            fig = plot.ax.get_figure()
        except SystemExit:
            notice.error("mdap stopped: the selected plot mode is incompatible with the data "
                         "dimensionality. Try a different plot_mode or adjust the X/Y/Z files.")
            return
        except Exception as exc:
            notice.error(f"Plotting failed: {exc}")
            return

        st.pyplot(fig, use_container_width=True)
        figure_download_button(fig, "md", "mdap_plot.png")
    plt.close(fig)

    # ---- equivalent Python ------------------------------------------------ #
    with st.expander("Equivalent Python", expanded=False):
        st.code(
            "import mdap\n\n"
            "X, Y, Z = mdap.Pdist(\n"
            + "".join(f"    {k}={v!r},\n" for k, v in pdist_kwargs.items())
            + ").pdist()\n\n"
            "mdap.Plot(X=X, Y=Y, Z=Z,\n"
            + "".join(f"    {k}={v!r},\n" for k, v in plot_kwargs.items())
            + ").plot()",
            language="python",
        )
