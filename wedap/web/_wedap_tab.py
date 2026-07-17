"""
wedap tab: WESTPA ``west.h5`` probability distributions + plots.

Mirrors the ``wedap`` CLI (matplotlib style, axis/cbar labels, argument
triaging) so plots look like their command-line equivalents. Exposes the pdist
options, plot modes (1D/2D/3D), formatting, plot tracing, and gif making.
"""
import base64
import io
import os
import tempfile

import h5py
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import wedap
from wedap.h5_gif import make_gif

from ._common import (
    MODE_DIM, PLOT_MODES, P_UNITS, P_UNITS_LABELS,
    parse_floats, parse_lim, parse_ints,
    apply_style_safe, run_postprocess, section,
    plot_formatting_controls, postprocess_controls, figure_download_button,
)

DATA_TYPES = ["evolution", "average", "instant"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def list_datasets(h5_path, first_iter=1):
    """Return (pcoord_options, aux_options) available in a west.h5 file."""
    aux = []
    try:
        with h5py.File(h5_path, "r") as f:
            grp = f.get(f"iterations/iter_{first_iter:08d}/auxdata")
            if grp is not None:
                aux = sorted(grp.keys())
    except (OSError, KeyError):
        pass
    return ["pcoord"], aux


def validate_combo(data_type, Yname, Zname, plot_mode):
    """
    Mirror the CLI's dimensionality triaging. Returns an error string describing
    why the current selection can't be plotted, or None if the combo is valid.
    """
    dim = MODE_DIM[plot_mode]
    if data_type == "evolution":
        if dim == 1:
            return ("Evolution plots are 2D (progress coordinate vs WE iteration). "
                    "Use a 2D mode like 'hist' or 'contour', or switch data_type to "
                    "'average'/'instant' for a 1D distribution.")
        if dim == 3:
            return "3D plot modes need data_type 'average' or 'instant' with a Zname."
        return None

    # average / instant
    is_1d = Yname is None and Zname is None
    if dim == 3:
        if Yname is None or Zname is None:
            return ("3D plot modes ('scatter3d'/'hexbin3d') plot 3 datasets: set both a "
                    "Yname and a Zname (e.g. pcoord i0 / i1 / i2 for 3 progress coords).")
        return None
    if is_1d and dim == 2:
        return ("This is a 1D distribution (average/instant with no Yname). Use plot "
                "mode 'line' or 'bar', or pick a Yname to make a 2D distribution.")
    if not is_1d and dim == 1:
        return ("You selected a Yname (2D distribution) but a 1D plot mode. Use 'hist' "
                "or a contour mode, or clear Yname for a 1D 'line'/'bar' plot.")
    return None


def default_ylabel(data_type, Yname, Yindex, p_units):
    if data_type == "evolution":
        return "WE Iteration"
    if Yname:
        return f"{Yname} i{Yindex}"
    # 1D distribution: the y-axis is the probability axis, so label it with the
    # p_units label (same as the colorbar for 2D plots)
    return P_UNITS_LABELS.get(p_units, "P(x)")


@st.cache_data(show_spinner="Computing probability distribution...")
def compute_pdist(h5_path, pdist_kwargs):
    """Run H5_Pdist and return (X, Y, Z). Cached on file path + pdist args."""
    X, Y, Z = wedap.H5_Pdist(h5=h5_path, **pdist_kwargs).pdist()
    return X, Y, Z


@st.cache_data(show_spinner=False)
def get_last_iter(h5_path):
    """
    Resolved final WE iteration for the file, read from the pdist object's
    ``last_iter`` attribute (i.e. west_current_iteration - 1).
    """
    # data_type is required by __init__ but irrelevant here (we don't run pdist)
    pdist = wedap.H5_Pdist(h5=h5_path, data_type="evolution", no_pbar=True)
    last = int(pdist.last_iter)
    try:
        pdist.h5.close()
    except Exception:
        pass
    return last


# --------------------------------------------------------------------------- #
# render
# --------------------------------------------------------------------------- #
def render_wedap():
    st.subheader("wedap — WESTPA H5 probability distributions & plots")
    st.caption("Interactive front-end for plotting WESTPA west.h5 files "
               "(a browser-based alternative to the `wedap` CLI).")

    # ---- data source (collapses to a slim summary once a file is loaded) --- #
    # `loaded` (last run's resolved file) drives the label; `_has` is derived from
    # the *current* widget selection so the panel doesn't snap shut mid-edit when
    # switching sources
    loaded = st.session_state.get("wd_loaded")
    _src = st.session_state.get("wd_source", "Server path")
    _has = (_src == "Example (p53.h5)")
    _has = _has or (_src == "Server path" and os.path.isfile(st.session_state.get("wd_path", "") or ""))
    _has = _has or (_src == "Upload file" and st.session_state.get("wd_upload") is not None)
    with st.expander(f"Data source — {loaded}" if loaded else "Data source",
                     expanded=not _has):
        source = st.radio(
            "Input mode",
            ["Server path", "Upload file", "Example (p53.h5)"],
            key="wd_source",
            help="Large west.h5 files should use a server-side path; uploads are "
                 "held in memory and best for small files.",
        )

        h5_path = None
        if source == "Server path":
            h5_path = st.text_input("Path to west.h5 file", value="", key="wd_path")
            if h5_path and not os.path.isfile(h5_path):
                st.warning("File not found at that path.")
                h5_path = None
        elif source == "Upload file":
            upload = st.file_uploader("west.h5 file", type=["h5", "hdf5"], key="wd_upload")
            if upload is not None:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".h5")
                tmp.write(upload.getbuffer())
                tmp.flush()
                h5_path = tmp.name
        else:  # packaged example
            example = os.path.join(os.path.dirname(wedap.__file__), "data", "p53.h5")
            h5_path = example if os.path.isfile(example) else None
            if h5_path is None:
                st.warning("Example p53.h5 not found in the installed package.")

    # remember the loaded file so the panel above shows it and collapses next run
    st.session_state["wd_loaded"] = os.path.basename(h5_path) if h5_path else None

    if not h5_path:
        st.info("Select a data source above to begin.")
        return

    pcoords, aux = list_datasets(h5_path)
    dataset_options = pcoords + aux

    # ---- pdist + plot options (left) beside the plot (right) -------------- #
    opts_col, plot_col = st.columns([1, 1])
    with opts_col, st.container(border=True):
        section("Pdist options")
        c1, c2 = st.columns(2)
        data_type = c1.selectbox("data_type", DATA_TYPES, index=0, key="wd_dt")
        p_units = c2.selectbox("p_units", P_UNITS, index=0, key="wd_pu")

        section("Plot mode")
        plot_mode = st.selectbox("plot_mode", PLOT_MODES, index=0, key="wd_pm")
        dim = MODE_DIM[plot_mode]

        section("Axes")
        c1, c2 = st.columns(2)
        Xname = c1.selectbox("Xname", dataset_options, index=0, key="wd_xn")
        Xindex = c2.number_input("Xindex", min_value=0, value=0, step=1, key="wd_xi")

        # Y available for average/instant; needed for 2D and 3D
        use_y = data_type != "evolution"
        Yname, Yindex = None, 0
        if use_y:
            c1, c2 = st.columns(2)
            y_sel = c1.selectbox("Yname", ["(none)"] + dataset_options, index=0, key="wd_yn")
            Yname = None if y_sel == "(none)" else y_sel
            Yindex = c2.number_input("Yindex", min_value=0, value=0, step=1, key="wd_yi")

        # Z available for average/instant 3D plots (3rd pcoord/aux as marker color)
        Zname, Zindex = None, 0
        if use_y and dim == 3:
            c1, c2 = st.columns(2)
            z_sel = c1.selectbox("Zname", ["(none)"] + dataset_options, index=0, key="wd_zn")
            Zname = None if z_sel == "(none)" else z_sel
            Zindex = c2.number_input("Zindex", min_value=0, value=0, step=1, key="wd_zi")

        section("Iterations & bins")
        c1, c2 = st.columns(2)
        first_iter = c1.number_input("first_iter", min_value=1, value=1, step=1, key="wd_fi")
        last_iter = c2.number_input("last_iter (0=all)", min_value=0, value=0, step=1, key="wd_li")
        c1, c2 = st.columns(2)
        bins = c1.number_input("bins", min_value=10, max_value=1000, value=100, step=10, key="wd_bins")
        step_iter = c2.number_input("step_iter", min_value=1, value=1, step=1, key="wd_step")

        # ---- shared formatting block -------------------------------------- #
        default_xlabel = f"{Xname} i{Xindex}"
        default_cbar = f"{Zname} i{Zindex}" if (dim == 3 and Zname) else ""
        plot_kwargs, meta = plot_formatting_controls(
            "wd", dim, plot_mode,
            default_xlabel=default_xlabel,
            default_ylabel=default_ylabel(data_type, Yname, Yindex, p_units),
            default_cbar=default_cbar,
            show_weighted=True,
        )
        style = meta["style"]

        # ---- tracing ------------------------------------------------------ #
        with st.expander("Plot tracing", expanded=False):
            trace_mode = st.radio("Trace", ["none", "by iter,seg", "by X,Y value"],
                                  horizontal=True, key="wd_trace_mode")
            trace_seg, trace_val = None, None
            if trace_mode == "by iter,seg":
                c1, c2 = st.columns(2)
                ti = c1.number_input("iteration", min_value=1, value=1, step=1, key="wd_ti")
                ts = c2.number_input("segment", min_value=0, value=0, step=1, key="wd_ts")
                trace_seg = (int(ti), int(ts))
            elif trace_mode == "by X,Y value":
                c1, c2 = st.columns(2)
                tx = c1.number_input("X value", value=0.0, key="wd_tx")
                tyv = c2.number_input("Y value", value=0.0, key="wd_tyv")
                trace_val = (float(tx), float(tyv))
            c1, c2 = st.columns(2)
            trace_color = c1.text_input("trace color", value="red", key="wd_tc")
            mark_points = c2.checkbox("mark start/end", value=False, key="wd_mp")

        # ---- advanced pdist ---------------------------------------------- #
        with st.expander("Advanced pdist", expanded=False):
            c1, c2 = st.columns(2)
            succ_only = c1.checkbox("succ_only", value=False, key="wd_succ",
                                    help="Only use successful trajectories.")
            skip_basis = parse_ints(
                c2.text_input("skip_basis", value="", key="wd_skip", help="e.g. 0 1 1"))
            c1, c2 = st.columns(2)
            histrange_x = parse_lim(c1.text_input("histrange_x (lo,hi)", value="", key="wd_hrx"))
            histrange_y = parse_lim(c2.text_input("histrange_y (lo,hi)", value="", key="wd_hry"))
            contour_levels = parse_floats(
                st.text_input("contour_levels", value="", key="wd_cl",
                              help="Explicit contour levels, comma-separated."))
            # 4D projection (scatter3d with an extra dataset as the cbar)
            Cname, Cindex, proj4d = None, 0, False
            if dim == 3:
                c1, c2 = st.columns(2)
                c_sel = c1.selectbox("Cname (4D cbar)", ["(none)"] + dataset_options, key="wd_cn")
                Cname = None if c_sel == "(none)" else c_sel
                Cindex = c2.number_input("Cindex", min_value=0, value=0, step=1, key="wd_ci4")
                proj4d = st.checkbox("proj4d", value=False, key="wd_proj4d",
                                     help="4D scatter; needs Cname + scatter3d.")

        # ---- postprocessing ---------------------------------------------- #
        pp_mode, pp_path, pp_code, pp_func = postprocess_controls("wd")

    # ---- validate the selection (message shown in the plot column below) --- #
    err = validate_combo(data_type, Yname, Zname, plot_mode)

    # ---- assemble kwargs -------------------------------------------------- #
    pdist_kwargs = dict(
        data_type=data_type,
        Xname=Xname, Xindex=int(Xindex),
        Yname=Yname, Yindex=int(Yindex),
        first_iter=int(first_iter),
        last_iter=None if last_iter == 0 else int(last_iter),
        step_iter=int(step_iter),
        bins=(int(bins), int(bins)),
        p_units=p_units,
        T=int(meta["T"]),
        weighted=bool(meta["weighted"]),
        succ_only=bool(succ_only),
        no_pbar=True,
    )
    if Zname is not None:
        pdist_kwargs["Zname"] = Zname
        pdist_kwargs["Zindex"] = int(Zindex)
    if Cname is not None:
        pdist_kwargs["Cname"] = Cname
        pdist_kwargs["Cindex"] = int(Cindex)
    if skip_basis is not None:
        pdist_kwargs["skip_basis"] = skip_basis
    if histrange_x is not None:
        pdist_kwargs["histrange_x"] = histrange_x
    if histrange_y is not None:
        pdist_kwargs["histrange_y"] = histrange_y

    # p_units belongs in both pdist and plot kwargs; proj4d added here
    plot_kwargs["p_units"] = p_units
    plot_kwargs["proj4d"] = bool(proj4d)
    if contour_levels is not None:
        plot_kwargs["contour_levels"] = contour_levels

    tracing = trace_seg is not None or trace_val is not None
    # tracing and 4D projection need the h5-backed object (they read the file);
    # jointplot must build its own raw (unnormalized) pdist and re-normalize
    # internally, so it can't use the already-normalized cached arrays
    needs_h5 = tracing or proj4d or meta["jointplot"]

    # merge so shared keys (e.g. p_units) aren't passed twice to the h5-backed call
    all_kwargs = {**pdist_kwargs, **plot_kwargs}

    # apply the matplotlib style just like the CLI (global rcParams)
    apply_style_safe(style, "wedap")

    # ---- compute + render, all inside the sticky plot column -------------- #
    # keeping the notice (warnings/errors) and the compute spinner in this
    # container means they stay pinned right above the plot rather than scrolling
    # away up the page past the options
    with plot_col, st.container(key="plot_sticky_wd"):
        notice = st.container()
        if err:
            notice.warning(f"⚠️ {err}")
            return

        try:
            if needs_h5:
                plot = wedap.H5_Plot(h5=h5_path, **all_kwargs)
                plot.plot()
                if trace_seg is not None:
                    plot.plot_trace(trace_seg, color=trace_color, ax=plot.ax,
                                    mark_points=mark_points)
                elif trace_val is not None:
                    iseg = plot.find_iter_seg_from_xy_vals(trace_val[0], trace_val[1])
                    plot.plot_trace(iseg, color=trace_color, ax=plot.ax,
                                    mark_points=mark_points)
            else:
                # compute_pdist's "Computing probability distribution..." spinner
                # renders here (in the plot column) since that's where it's called
                X, Y, Z = compute_pdist(h5_path, pdist_kwargs)
                plot = wedap.H5_Plot(X=X, Y=Y, Z=Z, **plot_kwargs)
                plot.plot()

            # optional user postprocessing on the finished plot
            if pp_mode != "none":
                try:
                    run_postprocess(plot, "file" if pp_mode == "file path" else "inline",
                                    pp_path, pp_code, pp_func, extra_ns={"wedap": wedap})
                except Exception as exc:
                    notice.warning(f"Postprocessing failed: {exc}")

            # resolve the real Figure: jointplot sets plot.fig to a mosaic dict of
            # axes (not a Figure), so pull the parent figure from the main axes
            fig = plot.ax.get_figure()
        except SystemExit:
            notice.error("wedap stopped: the selected plot mode is incompatible with this "
                         "dataset's dimensionality. Try a different plot_mode, or add/remove "
                         "a Yname (see the guidance for 1D vs 2D plots).")
            return
        except Exception as exc:
            notice.error(f"Plotting failed: {exc}")
            return

        st.pyplot(fig, use_container_width=True)
        figure_download_button(fig, "wd", "wedap_plot.png")
    plt.close(fig)

    # ---- gif making ------------------------------------------------------- #
    with st.expander("Make a GIF (average pdist over iterations)", expanded=False):
        st.caption("Loops over iterations building an average pdist per frame. "
                   "Best with 1D or 2D average plots.")
        resolved_last = get_last_iter(h5_path)
        c1, c2, c3 = st.columns(3)
        gif_first = c1.number_input("gif first_iter", min_value=1, value=1, step=1, key="wd_gf")
        gif_last = c2.number_input("gif last_iter (0=last)", min_value=0, value=0, step=1,
                                   key="wd_gl",
                                   help=f"0 uses the file's last iteration ({resolved_last}).")
        avg_plus = c3.number_input("avg_plus", min_value=0, value=10, step=1, key="wd_ap",
                                   help="Iterations added to each frame's average window.")
        duration = st.number_input("frame duration (ms)", min_value=1, value=50, step=10, key="wd_gd")
        if st.button("Generate GIF", key="wd_gifbtn"):
            # 0 means "use the actual last iteration"; cap the loop bound so the last
            # averaging window (iter + avg_plus) stays within available iterations
            if int(gif_last) == 0:
                gl = max(int(gif_first) + 1, resolved_last - int(avg_plus) + 1)
            else:
                gl = int(gif_last)
            gif_out = os.path.join(tempfile.gettempdir(), "wedap_web.gif")
            gif_kwargs = {k: v for k, v in all_kwargs.items()
                          if k not in ("first_iter", "last_iter", "step_iter",
                                       "jointplot", "proj3d", "proj4d")}
            try:
                with st.spinner("Building GIF..."):
                    apply_style_safe(style, "wedap")
                    make_gif(first_iter=int(gif_first), last_iter=gl,
                             step_iter=int(step_iter), avg_plus=int(avg_plus),
                             duration=int(duration), gif_out=gif_out,
                             h5=h5_path, **gif_kwargs)
                with open(gif_out, "rb") as fh:
                    gif_bytes = fh.read()
                # st.image shows only the first frame for GIFs; embed a data URI so
                # the animation plays (kept small)
                b64 = base64.b64encode(gif_bytes).decode()
                st.markdown(
                    f'<img src="data:image/gif;base64,{b64}" '
                    f'alt="wedap gif" style="max-width:420px; width:100%;">',
                    unsafe_allow_html=True,
                )
                st.download_button("Download GIF", data=gif_bytes,
                                   file_name="wedap.gif", mime="image/gif", key="wd_gifdl")
            except Exception as exc:
                st.error(f"GIF creation failed: {exc}")

    # ---- equivalent Python (below the plot, full width) ------------------- #
    with st.expander("Equivalent Python", expanded=False):
        st.code(
            "import wedap\n\n"
            "X, Y, Z = wedap.Pdist(\n"
            f"    h5={h5_path!r},\n"
            + "".join(f"    {k}={v!r},\n" for k, v in pdist_kwargs.items())
            + ").pdist()\n\n"
            "wedap.Plot(X, Y, Z,\n"
            + "".join(f"    {k}={v!r},\n" for k, v in plot_kwargs.items())
            + ").plot()",
            language="python",
        )
