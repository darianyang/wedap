"""
Streamlit front-end for wedap.

A self-contained browser UI over the core `wedap` (WESTPA H5) pdist + plotting
workflow, mirroring the CLI's behavior (matplotlib style, axis/cbar labels,
argument triaging) so plots look like their command-line equivalents. Exposes
the pdist options, plot modes (1D/2D/3D), formatting, plot tracing, and gif
making.

Run locally:
    streamlit run wedap/web/app.py
or (with the web extra installed):
    wedap-web

`mdap`/`wekap` tabs can be added the same way.

Notes
-----
* Real ``west.h5`` files can be very large (GB-scale), so the default input mode
  is a **server-side path** rather than an upload.
* The expensive pdist computation is cached with ``st.cache_data`` keyed on the
  file + pdist arguments, so display-only tweaks (cmap, limits, labels) do not
  recompute the histogram. Tracing and gif making use the h5-backed path (those
  methods read the file directly), so they bypass the cache.
"""
import base64
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
from wedap.utils import apply_style
from wedap.h5_gif import make_gif


# --------------------------------------------------------------------------- #
# plot-mode metadata (names must match wedap/command_line.py choices exactly)
# --------------------------------------------------------------------------- #
# dimensionality each plot mode expects
MODE_DIM = {
    "hist": 2, "hist_l": 2, "contour": 2, "contour_l": 2, "contour_f": 2,
    "line": 1, "bar": 1,
    "scatter3d": 3, "hexbin3d": 3,
}
PLOT_MODES = list(MODE_DIM)
P_UNITS = ["kT", "kcal", "raw", "raw_norm", "raw_norm_tot"]
DATA_TYPES = ["evolution", "average", "instant"]

# probability-units axis labels, matching H5_Plot.set_cbar_label(); used for the
# colorbar (2D) and for the y-axis of 1D line/bar plots (the probability axis)
P_UNITS_LABELS = {
    "kT": r"$-\ln\,P(x)$",
    "kcal": r"$-RT\ \ln\, P\ (kcal\ mol^{-1})$",
    "raw": "Counts",
    "raw_norm": "Normalized Counts",
    "raw_norm_tot": "Probability",
}


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


def default_xlabel(Xname, Xindex):
    return f"{Xname} i{Xindex}"


def default_ylabel(data_type, Yname, Yindex, p_units):
    if data_type == "evolution":
        return "WE Iteration"
    if Yname:
        return f"{Yname} i{Yindex}"
    # 1D distribution: the y-axis is the probability axis, so label it with the
    # p_units label (same as the colorbar for 2D plots)
    return P_UNITS_LABELS.get(p_units, "P(x)")


def parse_floats(text):
    """Parse a comma/space separated string into a list of floats, or None."""
    if not text or not text.strip():
        return None
    try:
        return [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        return None


def parse_lim(text):
    """Parse 'lo, hi' into a 2-list of floats, or None."""
    vals = parse_floats(text)
    if vals and len(vals) == 2:
        return vals
    return None


def parse_ints(text):
    """Parse a comma/space separated string into a list of ints, or None."""
    if not text or not text.strip():
        return None
    try:
        return [int(v) for v in text.replace(",", " ").split()]
    except ValueError:
        return None


def run_postprocess(plot, mode, path, code, func_name):
    """
    Execute a user postprocessing step against the current plot.

    ``plot``, ``ax``, ``fig``, ``plt``, ``np``, and ``wedap`` are placed in scope.
    In "file" mode the .py file at ``path`` is read; in "inline" mode ``code`` is
    used. If ``func_name`` is given and defined after exec, it is then called with
    no arguments (matching the CLI's ``module.function`` convention).
    """
    ns = {"plot": plot, "ax": plot.ax, "fig": plot.fig,
          "plt": plt, "np": np, "wedap": wedap}
    src = code
    if mode == "file":
        with open(path) as fh:
            src = fh.read()
    exec(src, ns)  # local analysis tool: user-provided code, like the CLI's postprocess
    name = (func_name or "").strip()
    if name and name in ns:
        ns[name]()


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
# app
# --------------------------------------------------------------------------- #
def main():
    st.set_page_config(page_title="wedap", page_icon="📊", layout="wide")

    # widen the sidebar so the two-column option layout has room to breathe
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] { min-width: 430px; max-width: 560px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

    pcoords, aux = list_datasets(h5_path)
    dataset_options = pcoords + aux

    # ---- pdist + plot options (two-column sidebar) ------------------------ #
    with st.sidebar:
        st.header("Pdist options")
        c1, c2 = st.columns(2)
        data_type = c1.selectbox("data_type", DATA_TYPES, index=0)
        p_units = c2.selectbox("p_units", P_UNITS, index=0)

        st.header("Plot mode")
        plot_mode = st.selectbox("plot_mode", PLOT_MODES, index=0)
        dim = MODE_DIM[plot_mode]

        st.header("Axes")
        c1, c2 = st.columns(2)
        Xname = c1.selectbox("Xname", dataset_options, index=0)
        Xindex = c2.number_input("Xindex", min_value=0, value=0, step=1)

        # Y available for average/instant; needed for 2D and 3D
        use_y = data_type != "evolution"
        Yname, Yindex = None, 0
        if use_y:
            c1, c2 = st.columns(2)
            y_sel = c1.selectbox("Yname", ["(none)"] + dataset_options, index=0)
            Yname = None if y_sel == "(none)" else y_sel
            Yindex = c2.number_input("Yindex", min_value=0, value=0, step=1)

        # Z available for average/instant 3D plots (3rd pcoord/aux as marker color)
        Zname, Zindex = None, 0
        if use_y and dim == 3:
            c1, c2 = st.columns(2)
            z_sel = c1.selectbox("Zname", ["(none)"] + dataset_options, index=0)
            Zname = None if z_sel == "(none)" else z_sel
            Zindex = c2.number_input("Zindex", min_value=0, value=0, step=1)

        st.header("Iterations & bins")
        c1, c2 = st.columns(2)
        first_iter = c1.number_input("first_iter", min_value=1, value=1, step=1)
        last_iter = c2.number_input("last_iter (0=all)", min_value=0, value=0, step=1)
        c1, c2 = st.columns(2)
        bins = c1.number_input("bins", min_value=10, max_value=1000, value=100, step=10)
        step_iter = c2.number_input("step_iter", min_value=1, value=1, step=1)

        # ---- formatting (collapsed by default) ---------------------------- #
        with st.expander("Plot formatting", expanded=False):
            c1, c2 = st.columns(2)
            style = c1.text_input("style", value="default",
                                  help="'default' wedap style, 'None' for base mpl, "
                                       "a named mpl style (e.g. ggplot), or a path.")
            cmap = c2.text_input("cmap", value="viridis")

            c1, c2 = st.columns(2)
            color = c1.text_input("color", value="",
                                  help="1D lines / contour lines / traces, e.g. 'k'.")
            alpha = c2.slider("alpha", 0.0, 1.0, 1.0, 0.05)

            c1, c2 = st.columns(2)
            xlabel = c1.text_input("xlabel", value=default_xlabel(Xname, Xindex))
            ylabel = c2.text_input(
                "ylabel", value=default_ylabel(data_type, Yname, Yindex, p_units))

            c1, c2 = st.columns(2)
            # for 3D, the cbar/z-axis defaults to the Z dataset name (like the CLI)
            default_cbar = f"{Zname} i{Zindex}" if (dim == 3 and Zname) else ""
            cbar_label = c1.text_input("cbar_label", value=default_cbar,
                                       help="Blank = auto from p_units (2D) or Zname (3D).")
            zlabel = c2.text_input("zlabel", value="", help="3D z-axis label.")

            c1, c2 = st.columns(2)
            title = c1.text_input("title", value="")
            suptitle = c2.text_input("suptitle", value="")

            c1, c2 = st.columns(2)
            xlim = c1.text_input("xlim (lo,hi)", value="")
            ylim = c2.text_input("ylim (lo,hi)", value="")

            c1, c2 = st.columns(2)
            p_min = c1.text_input("p_min", value="")
            p_max = c2.text_input("p_max", value="")

            c1, c2 = st.columns(2)
            axvline = c1.text_input("axvline(s)", value="", help="Comma-separated x values.")
            axhline = c2.text_input("axhline(s)", value="", help="Comma-separated y values.")

            c1, c2 = st.columns(2)
            grid = c1.checkbox("grid", value=False)
            jointplot = c2.checkbox("jointplot", value=False,
                                    help="Add marginal plots (2D only).")

            c1, c2 = st.columns(2)
            weighted = c1.checkbox("weighted", value=True, help="Use WE weights.")
            T = c2.number_input("T (K, for kcal)", min_value=1, value=298)

            sl = st.text_input("smoothing_level", value="",
                               help="Gaussian smoothing sigma for hist/contour.")
            smoothing_level = float(sl) if sl.strip() else None

            # mode-specific extras
            contour_interval = 1.0
            linewidth, linestyle = None, "-"
            scatter_interval, scatter_s, hexbin_grid = 10, 1.0, 100
            proj3d = False
            if dim == 2 and plot_mode != "hist":
                contour_interval = st.number_input(
                    "contour_interval", min_value=0.01, value=1.0, step=0.5)
            if plot_mode in ("line", "bar", "contour", "contour_l", "hist_l"):
                c1, c2 = st.columns(2)
                lw = c1.text_input("linewidth", value="")
                linewidth = float(lw) if lw.strip() else None
                linestyle = c2.text_input("linestyle", value="-") or "-"
            if dim == 3:
                proj3d = st.checkbox("proj3d", value=False,
                                     help="Render as a 3D projection.")
            if plot_mode == "scatter3d":
                c1, c2 = st.columns(2)
                scatter_interval = c1.number_input("scatter_interval", min_value=1, value=10)
                scatter_s = c2.number_input("scatter_s", min_value=0.1, value=1.0)
            if plot_mode == "hexbin3d":
                hexbin_grid = st.number_input("hexbin_grid", min_value=10, value=100)

        # ---- tracing ------------------------------------------------------ #
        with st.expander("Plot tracing", expanded=False):
            trace_mode = st.radio("Trace", ["none", "by iter,seg", "by X,Y value"],
                                  horizontal=True)
            trace_seg, trace_val = None, None
            if trace_mode == "by iter,seg":
                c1, c2 = st.columns(2)
                ti = c1.number_input("iteration", min_value=1, value=1, step=1)
                ts = c2.number_input("segment", min_value=0, value=0, step=1)
                trace_seg = (int(ti), int(ts))
            elif trace_mode == "by X,Y value":
                c1, c2 = st.columns(2)
                tx = c1.number_input("X value", value=0.0)
                tyv = c2.number_input("Y value", value=0.0)
                trace_val = (float(tx), float(tyv))
            c1, c2 = st.columns(2)
            trace_color = c1.text_input("trace color", value="red")
            mark_points = c2.checkbox("mark start/end", value=False)

        # ---- advanced pdist ---------------------------------------------- #
        with st.expander("Advanced pdist", expanded=False):
            c1, c2 = st.columns(2)
            succ_only = c1.checkbox("succ_only", value=False,
                                    help="Only use successful trajectories.")
            skip_basis = parse_ints(
                c2.text_input("skip_basis", value="", help="e.g. 0 1 1"))
            c1, c2 = st.columns(2)
            histrange_x = parse_lim(c1.text_input("histrange_x (lo,hi)", value=""))
            histrange_y = parse_lim(c2.text_input("histrange_y (lo,hi)", value=""))
            contour_levels = parse_floats(
                st.text_input("contour_levels", value="",
                              help="Explicit contour levels, comma-separated."))
            # 4D projection (scatter3d with an extra dataset as the cbar)
            Cname, Cindex, proj4d = None, 0, False
            if dim == 3:
                c1, c2 = st.columns(2)
                c_sel = c1.selectbox("Cname (4D cbar)", ["(none)"] + dataset_options)
                Cname = None if c_sel == "(none)" else c_sel
                Cindex = c2.number_input("Cindex", min_value=0, value=0, step=1)
                proj4d = st.checkbox("proj4d", value=False,
                                     help="4D scatter; needs Cname + scatter3d.")

        # ---- postprocessing ---------------------------------------------- #
        with st.expander("Postprocessing", expanded=False):
            pp_mode = st.radio("Source", ["none", "file path", "inline code"],
                               horizontal=True)
            pp_path, pp_code, pp_func = "", "", ""
            if pp_mode == "file path":
                pp_path = st.text_input("Path to .py file")
                pp_func = st.text_input("Function to call", value="",
                                        help="Called with no args after import "
                                             "(operates on the current plot).")
            elif pp_mode == "inline code":
                pp_code = st.text_area(
                    "Python (plot, ax, fig, plt, np, wedap in scope)",
                    value="ax.axvline(3, color='k', ls='--')", height=120)
                pp_func = st.text_input("Function to call (optional)", value="")

    # ---- validate the selection before doing any work --------------------- #
    err = validate_combo(data_type, Yname, Zname, plot_mode)
    if err:
        st.warning(f"⚠️ {err}")
        st.stop()

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
        T=int(T),
        weighted=bool(weighted),
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

    plot_kwargs = dict(
        plot_mode=plot_mode,
        cmap=cmap or None,
        p_units=p_units,
        xlabel=xlabel or None,
        ylabel=ylabel or None,
        contour_interval=contour_interval,
        linewidth=linewidth,
        linestyle=linestyle,
        scatter_interval=int(scatter_interval),
        scatter_s=float(scatter_s),
        hexbin_grid=int(hexbin_grid),
        jointplot=bool(jointplot),
        proj3d=bool(proj3d),
        proj4d=bool(proj4d),
    )
    if contour_levels is not None:
        plot_kwargs["contour_levels"] = contour_levels
    if smoothing_level is not None:
        plot_kwargs["smoothing_level"] = smoothing_level
    if color.strip():
        plot_kwargs["color"] = color.strip()
    if alpha < 1.0:
        plot_kwargs["alpha"] = alpha
    if title.strip():
        plot_kwargs["title"] = title
    if suptitle.strip():
        plot_kwargs["suptitle"] = suptitle
    if zlabel.strip():
        plot_kwargs["zlabel"] = zlabel
    if cbar_label.strip():
        plot_kwargs["cbar_label"] = cbar_label
    if grid:
        plot_kwargs["grid"] = True
    if parse_lim(xlim):
        plot_kwargs["xlim"] = parse_lim(xlim)
    if parse_lim(ylim):
        plot_kwargs["ylim"] = parse_lim(ylim)
    if p_min.strip():
        plot_kwargs["p_min"] = float(p_min)
    if p_max.strip():
        plot_kwargs["p_max"] = float(p_max)
    if parse_floats(axvline):
        plot_kwargs["axvline"] = parse_floats(axvline)
    if parse_floats(axhline):
        plot_kwargs["axhline"] = parse_floats(axhline)

    tracing = trace_seg is not None or trace_val is not None
    # tracing and 4D projection need the h5-backed object (they read the file);
    # jointplot must build its own raw (unnormalized) pdist and re-normalize
    # internally, so it can't use the already-normalized cached arrays; everything
    # else can use the cached pdist arrays for a fast re-render
    needs_h5 = tracing or proj4d or jointplot

    # merge so shared keys (e.g. p_units) aren't passed twice to the h5-backed call
    all_kwargs = {**pdist_kwargs, **plot_kwargs}

    # apply the matplotlib style just like the CLI (global rcParams)
    try:
        apply_style(style, "wedap")
    except (OSError, ValueError) as exc:
        st.warning(f"Could not apply style '{style}': {exc}. Using current settings.")

    # ---- compute + render ------------------------------------------------- #
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
            X, Y, Z = compute_pdist(h5_path, pdist_kwargs)
            plot = wedap.H5_Plot(X=X, Y=Y, Z=Z, **plot_kwargs)
            plot.plot()

        # optional user postprocessing on the finished plot
        if pp_mode != "none":
            try:
                run_postprocess(plot, "file" if pp_mode == "file path" else "inline",
                                pp_path, pp_code, pp_func)
            except Exception as exc:
                st.warning(f"Postprocessing failed: {exc}")

        # resolve the real Figure: jointplot sets plot.fig to a mosaic dict of
        # axes (not a Figure), so pull the parent figure from the main axes
        fig = plot.ax.get_figure()
    except SystemExit:
        # wedap calls sys.exit() on an incompatible mode/data combination
        st.error("wedap stopped: the selected plot mode is incompatible with this "
                 "dataset's dimensionality. Try a different plot_mode, or add/remove "
                 "a Yname (see the guidance for 1D vs 2D plots).")
        st.stop()
    except Exception as exc:
        st.error(f"Plotting failed: {exc}")
        st.stop()

    # render at a modest size (constrain to ~60% of the page width)
    plot_col, _ = st.columns([3, 2])
    with plot_col:
        st.pyplot(fig, use_container_width=True)

        # download the current figure as a PNG
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        st.download_button(
            "Download PNG",
            data=buf.getvalue(),
            file_name="wedap_plot.png",
            mime="image/png",
        )
    plt.close(fig)

    # ---- gif making ------------------------------------------------------- #
    with st.expander("Make a GIF (average pdist over iterations)", expanded=False):
        st.caption("Loops over iterations building an average pdist per frame. "
                   "Best with 1D or 2D average plots.")
        # actual final iteration for the file (from the pdist object)
        resolved_last = get_last_iter(h5_path)
        c1, c2, c3 = st.columns(3)
        gif_first = c1.number_input("gif first_iter", min_value=1, value=1, step=1)
        gif_last = c2.number_input("gif last_iter (0=last)", min_value=0, value=0, step=1,
                                   help=f"0 uses the file's last iteration ({resolved_last}).")
        avg_plus = c3.number_input("avg_plus", min_value=0, value=10, step=1,
                                   help="Iterations added to each frame's average window.")
        duration = st.number_input("frame duration (ms)", min_value=1, value=50, step=10)
        if st.button("Generate GIF"):
            # 0 means "use the actual last iteration" (like the pdist convention);
            # cap the loop bound so the last averaging window (iter + avg_plus)
            # stays within the file's available iterations
            if int(gif_last) == 0:
                gl = max(int(gif_first) + 1, resolved_last - int(avg_plus) + 1)
            else:
                gl = int(gif_last)
            gif_out = os.path.join(tempfile.gettempdir(), "wedap_web.gif")
            gif_kwargs = {k: v for k, v in {**pdist_kwargs, **plot_kwargs}.items()
                          if k not in ("first_iter", "last_iter", "step_iter",
                                       "jointplot", "proj3d", "proj4d")}
            try:
                with st.spinner("Building GIF..."):
                    # re-apply the style so the gif frames pick up the wedap
                    # rcParams (fonts, line widths, savefig.dpi) like the main plot
                    apply_style(style, "wedap")
                    make_gif(first_iter=int(gif_first), last_iter=gl,
                             step_iter=int(step_iter), avg_plus=int(avg_plus),
                             duration=int(duration), gif_out=gif_out,
                             h5=h5_path, **gif_kwargs)
                with open(gif_out, "rb") as fh:
                    gif_bytes = fh.read()
                # st.image shows only the first frame for GIFs; embed a data URI
                # in an <img> tag so the animation actually plays (kept small)
                b64 = base64.b64encode(gif_bytes).decode()
                st.markdown(
                    f'<img src="data:image/gif;base64,{b64}" '
                    f'alt="wedap gif" style="max-width:420px; width:100%;">',
                    unsafe_allow_html=True,
                )
                st.download_button("Download GIF", data=gif_bytes,
                                   file_name="wedap.gif", mime="image/gif")
            except Exception as exc:
                st.error(f"GIF creation failed: {exc}")

    # ---- equivalent Python (below the plot, full width) ------------------- #
    st.subheader("Equivalent Python")
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


if __name__ == "__main__":
    main()
