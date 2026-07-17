"""
Shared helpers and reusable option widgets for the wedap/mdap/wekap web app.

``wedap`` and ``mdap`` share the same underlying plotting class (``H5_Plot``), so
the plot-mode metadata, probability-unit labels, small parsers, and the big
"Plot formatting" control block all live here and are reused by both tabs.
``wekap`` (rate/kinetics plots) reuses only the small parsers and the
postprocessing block.
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from wedap.utils import apply_style


# --------------------------------------------------------------------------- #
# plot-mode metadata (names must match the CLI choices exactly)
# --------------------------------------------------------------------------- #
# dimensionality each plot mode expects
MODE_DIM = {
    "hist": 2, "hist_l": 2, "contour": 2, "contour_l": 2, "contour_f": 2,
    "line": 1, "bar": 1,
    "scatter3d": 3, "hexbin3d": 3,
}
PLOT_MODES = list(MODE_DIM)
P_UNITS = ["kT", "kcal", "raw", "raw_norm", "raw_norm_tot"]

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
# small parsers
# --------------------------------------------------------------------------- #
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


def section(label):
    """A compact section header (uses far less vertical space than st.header)."""
    st.markdown(f"**{label}**")


def dynamic_text_input(container, label, key, default, help=None):
    """
    A keyed ``text_input`` whose default value tracks ``default`` across reruns.

    Keyed Streamlit widgets store their value in ``session_state`` and ignore the
    ``value=`` argument after first render, which would freeze a computed default
    (e.g. the y-label that depends on ``data_type``/``Yname``). This mirrors the
    original keyless behavior: the field follows the computed default until the
    user edits it, and re-syncs whenever the computed default itself changes.

    The widget's own value is also cleared from ``session_state`` when its tool is
    not the active one (Streamlit drops state for widgets it didn't render this
    run), so we re-seed whenever the key is missing too — otherwise returning to a
    tab would show a blank field.
    """
    auto_key = f"{key}__auto"
    if key not in st.session_state or st.session_state.get(auto_key) != default:
        st.session_state[key] = default
    st.session_state[auto_key] = default
    return container.text_input(label, key=key, help=help)


def apply_style_safe(style, package):
    """Apply a matplotlib style like the CLI, warning (not raising) on failure."""
    try:
        apply_style(style, package)
    except (OSError, ValueError) as exc:
        st.warning(f"Could not apply style '{style}': {exc}. Using current settings.")


# --------------------------------------------------------------------------- #
# user postprocessing (shared by all three tabs)
# --------------------------------------------------------------------------- #
def run_postprocess(plot, mode, path, code, func_name, extra_ns=None):
    """
    Execute a user postprocessing step against the current plot object.

    ``plot``, ``ax``, ``fig``, ``plt``, and ``np`` are placed in scope, plus
    anything in ``extra_ns`` (e.g. ``{"wedap": wedap}``). In "file" mode the .py
    file at ``path`` is read; in "inline" mode ``code`` is used. If ``func_name``
    is given and defined after exec, it is then called with no arguments
    (matching the CLI's ``module.function`` convention).
    """
    ns = {"plot": plot, "ax": plot.ax, "fig": plot.fig, "plt": plt, "np": np}
    if extra_ns:
        ns.update(extra_ns)
    src = code
    if mode == "file":
        with open(path) as fh:
            src = fh.read()
    exec(src, ns)  # local analysis tool: user-provided code, like the CLI's postprocess
    name = (func_name or "").strip()
    if name and name in ns:
        ns[name]()


def postprocess_controls(prefix, default_inline="ax.axvline(3, color='k', ls='--')"):
    """
    Render the shared postprocessing expander.

    Returns ``(mode, path, code, func)`` where ``mode`` is one of
    "none"/"file path"/"inline code".
    """
    with st.expander("Postprocessing", expanded=False):
        mode = st.radio("Source", ["none", "file path", "inline code"],
                        horizontal=True, key=f"{prefix}_pp_mode")
        path, code, func = "", "", ""
        if mode == "file path":
            path = st.text_input("Path to .py file", key=f"{prefix}_pp_path")
            func = st.text_input("Function to call", value="", key=f"{prefix}_pp_func",
                                 help="Called with no args after import "
                                      "(operates on the current plot).")
        elif mode == "inline code":
            code = st.text_area("Python (plot, ax, fig, plt, np in scope)",
                                value=default_inline, height=120, key=f"{prefix}_pp_code")
            func = st.text_input("Function to call (optional)", value="",
                                 key=f"{prefix}_pp_func_inline")
    return mode, path, code, func


# --------------------------------------------------------------------------- #
# shared "Plot formatting" control block (wedap + mdap; both drive H5_Plot)
# --------------------------------------------------------------------------- #
def plot_formatting_controls(prefix, dim, plot_mode, default_xlabel="",
                             default_ylabel="", default_cbar="",
                             show_weighted=True):
    """
    Render the "Plot formatting" expander shared by the wedap and mdap tabs and
    return ``(plot_kwargs, meta)``.

    ``plot_kwargs`` are ready to pass to ``H5_Plot``/``MD_Plot`` (cmap, labels,
    limits, contour/scatter/hexbin extras, jointplot, proj3d, ...). ``meta`` holds
    values routed elsewhere: ``style`` (for apply_style), ``weighted`` and ``T``
    (pdist options), and the ``jointplot``/``proj3d`` flags (also echoed in
    plot_kwargs) so callers can branch on them.
    """
    plot_kwargs = {}
    meta = {}
    with st.expander("Plot formatting", expanded=False):
        c1, c2 = st.columns(2)
        meta["style"] = c1.text_input(
            "style", value="default", key=f"{prefix}_style",
            help="'default' style, 'None' for base mpl, a named mpl style "
                 "(e.g. ggplot), or a path.")
        cmap = c2.text_input("cmap", value="viridis", key=f"{prefix}_cmap")

        c1, c2 = st.columns(2)
        color = c1.text_input("color", value="", key=f"{prefix}_color",
                              help="1D lines / contour lines / traces, e.g. 'k'.")
        alpha = c2.slider("alpha", 0.0, 1.0, 1.0, 0.05, key=f"{prefix}_alpha")

        # xlabel/ylabel/cbar_label defaults are computed from the current selections
        # (data_type, Xname/Yname/Zname, p_units), so they must follow those changes
        c1, c2 = st.columns(2)
        xlabel = dynamic_text_input(c1, "xlabel", f"{prefix}_xlabel", default_xlabel)
        ylabel = dynamic_text_input(c2, "ylabel", f"{prefix}_ylabel", default_ylabel)

        c1, c2 = st.columns(2)
        cbar_label = dynamic_text_input(c1, "cbar_label", f"{prefix}_cbar", default_cbar,
                                        help="Blank = auto from p_units (2D) or Zname (3D).")
        zlabel = c2.text_input("zlabel", value="", key=f"{prefix}_zlabel",
                               help="3D z-axis label.")

        c1, c2 = st.columns(2)
        title = c1.text_input("title", value="", key=f"{prefix}_title")
        suptitle = c2.text_input("suptitle", value="", key=f"{prefix}_suptitle")

        c1, c2 = st.columns(2)
        xlim = c1.text_input("xlim (lo,hi)", value="", key=f"{prefix}_xlim")
        ylim = c2.text_input("ylim (lo,hi)", value="", key=f"{prefix}_ylim")

        c1, c2 = st.columns(2)
        p_min = c1.text_input("p_min", value="", key=f"{prefix}_pmin")
        p_max = c2.text_input("p_max", value="", key=f"{prefix}_pmax")

        c1, c2 = st.columns(2)
        axvline = c1.text_input("axvline(s)", value="", key=f"{prefix}_axvline",
                                help="Comma-separated x values.")
        axhline = c2.text_input("axhline(s)", value="", key=f"{prefix}_axhline",
                                help="Comma-separated y values.")

        c1, c2 = st.columns(2)
        grid = c1.checkbox("grid", value=False, key=f"{prefix}_grid")
        jointplot = c2.checkbox("jointplot", value=False, key=f"{prefix}_jointplot",
                                help="Add marginal plots (2D only).")

        if show_weighted:
            c1, c2 = st.columns(2)
            meta["weighted"] = c1.checkbox("weighted", value=True, key=f"{prefix}_weighted",
                                           help="Use WE weights.")
            meta["T"] = c2.number_input("T (K, for kcal)", min_value=1, value=298,
                                        key=f"{prefix}_T")
        else:
            meta["weighted"] = False
            meta["T"] = st.number_input("T (K, for kcal)", min_value=1, value=298,
                                        key=f"{prefix}_T")

        sl = st.text_input("smoothing_level", value="", key=f"{prefix}_sl",
                           help="Gaussian smoothing sigma for hist/contour.")
        smoothing_level = float(sl) if sl.strip() else None

        # mode-specific extras
        contour_interval = 1.0
        linewidth, linestyle = None, "-"
        scatter_interval, scatter_s, hexbin_grid = 10, 1.0, 100
        proj3d = False
        if dim == 2 and plot_mode != "hist":
            contour_interval = st.number_input("contour_interval", min_value=0.01,
                                               value=1.0, step=0.5, key=f"{prefix}_ci")
        if plot_mode in ("line", "bar", "contour", "contour_l", "hist_l"):
            c1, c2 = st.columns(2)
            lw = c1.text_input("linewidth", value="", key=f"{prefix}_lw")
            linewidth = float(lw) if lw.strip() else None
            linestyle = c2.text_input("linestyle", value="-", key=f"{prefix}_ls") or "-"
        if dim == 3:
            proj3d = st.checkbox("proj3d", value=False, key=f"{prefix}_proj3d",
                                 help="Render as a 3D projection.")
        if plot_mode == "scatter3d":
            c1, c2 = st.columns(2)
            scatter_interval = c1.number_input("scatter_interval", min_value=1, value=10,
                                               key=f"{prefix}_sci")
            scatter_s = c2.number_input("scatter_s", min_value=0.1, value=1.0,
                                        key=f"{prefix}_scs")
        if plot_mode == "hexbin3d":
            hexbin_grid = st.number_input("hexbin_grid", min_value=10, value=100,
                                          key=f"{prefix}_hbg")

    # assemble the H5_Plot-ready plot kwargs
    plot_kwargs.update(
        plot_mode=plot_mode,
        cmap=cmap or None,
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
    )
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

    meta["jointplot"] = bool(jointplot)
    meta["proj3d"] = bool(proj3d)
    return plot_kwargs, meta


def figure_download_button(fig, prefix, filename):
    """Render side-by-side 'Download PNG' (raster) and 'Download PDF' (vector) buttons."""
    import io
    import os

    png_buf = io.BytesIO()
    fig.savefig(png_buf, format="png", dpi=300, bbox_inches="tight")
    pdf_buf = io.BytesIO()
    fig.savefig(pdf_buf, format="pdf", bbox_inches="tight")

    stem = os.path.splitext(filename)[0]
    # narrow button columns + a wide spacer keep the two buttons adjacent rather
    # than spread across the full plot width
    c1, c2, _ = st.columns([1, 1, 3], gap="small")
    c1.download_button("Download PNG", data=png_buf.getvalue(), file_name=f"{stem}.png",
                       mime="image/png", key=f"{prefix}_dl_png", use_container_width=True)
    c2.download_button("Download PDF", data=pdf_buf.getvalue(), file_name=f"{stem}.pdf",
                       mime="application/pdf", key=f"{prefix}_dl_pdf", use_container_width=True)
