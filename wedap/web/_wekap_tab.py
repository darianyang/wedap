"""
wekap tab: WE kinetics — plot rates/MFPTs from a WESTPA ``direct.h5`` file.

Mirrors the ``wekap`` CLI: a single ``direct.h5`` (optionally with an
``assign.h5`` for labeled populations), tau/state/concentration options, the RED
duration-correction scheme, rate vs. MFPT units, and multi-file replicate
averaging with bootstrapped error.
"""
import os
import tempfile

import matplotlib.pyplot as plt
import streamlit as st

import wekap
from ._common import (
    parse_floats, parse_lim, apply_style_safe, run_postprocess, section,
    postprocess_controls, figure_download_button,
)


def _save_upload(upload, suffix=".h5"):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.getbuffer())
    tmp.flush()
    return tmp.name


def render_wekap():
    st.subheader("wekap — weighted ensemble kinetics (rates & MFPTs)")
    st.caption("Plot flux values from a WESTPA `direct.h5` file as rates or MFPTs "
               "(a browser-based alternative to the `wekap` CLI).")

    example_dir = os.path.join(os.path.dirname(wekap.__file__), "data")

    # `loaded` drives the label; `_has` (from the current widget selection) drives
    # the expanded state so the panel doesn't snap shut mid-edit when switching
    loaded = st.session_state.get("wk_loaded")
    _src = st.session_state.get("wk_source", "Server path")
    _has = (_src == "Example")
    _has = _has or (_src == "Server path" and os.path.isfile(st.session_state.get("wk_direct", "") or ""))
    _has = _has or (_src == "Upload" and st.session_state.get("wk_ud") is not None)
    with st.expander(f"Data source — {loaded}" if loaded else "Data source",
                     expanded=not _has):
        source = st.radio("Input mode", ["Server path", "Upload", "Example"],
                          key="wk_source")

        direct, assign, multi_direct = None, None, []
        if source == "Server path":
            direct = st.text_input("direct.h5 path", value="", key="wk_direct") or None
            if direct and not os.path.isfile(direct):
                st.warning("direct.h5 not found at that path.")
                direct = None
            assign = st.text_input("assign.h5 path (optional)", value="", key="wk_assign") or None
            extra = st.text_area("Extra direct.h5 files (one per line, optional)",
                                 value="", key="wk_multi",
                                 help="Add replicate direct.h5 paths to average with "
                                      "bootstrapped error via plot_multi_rates.")
            multi_direct = [p.strip() for p in extra.splitlines() if p.strip()]
        elif source == "Upload":
            ud = st.file_uploader("direct.h5", type=["h5"], key="wk_ud")
            ua = st.file_uploader("assign.h5 (optional)", type=["h5"], key="wk_ua")
            direct = _save_upload(ud) if ud else None
            assign = _save_upload(ua) if ua else None
        else:  # example
            ex = os.path.join(example_dir, "direct.h5")
            direct = ex if os.path.isfile(ex) else None
            st.caption("Example: bundled `direct.h5` (+ `assign.h5`).")
            if st.checkbox("Average 3 example replicates", value=False, key="wk_ex_multi"):
                multi_direct = [os.path.join(example_dir, f)
                                for f in ("direct.h5", "direct2.h5", "direct3.h5")]

    # remember the loaded file so the panel above shows it and collapses next run
    st.session_state["wk_loaded"] = os.path.basename(direct) if direct else None

    if not direct:
        st.info("Provide a direct.h5 file above to begin.")
        return

    opts_col, plot_col = st.columns([1, 1])
    with opts_col, st.container(border=True):
        section("Kinetics options")
        c1, c2 = st.columns(2)
        tau_ps = c1.number_input("tau (ps)", min_value=0.001, value=100.0, step=10.0,
                                 key="wk_tau", help="Resampling interval; converted to seconds.")
        state = c2.number_input("state", min_value=0, value=1, step=1, key="wk_state",
                                help="Target state for flux (0=A, 1=B).")
        c1, c2 = st.columns(2)
        flux_units = c1.selectbox("flux_units", ["rates", "mfpts"], index=0, key="wk_fu")
        x_units = c2.selectbox("x_units", ["iterations", "moltime", "agg"], index=0, key="wk_xu")
        c1, c2 = st.columns(2)
        statepop = c1.selectbox("statepop", ["direct", "assign"], index=0, key="wk_sp")
        concentration = c2.number_input("concentration (M)", min_value=0.0, value=1.0,
                                        step=1.0, key="wk_conc",
                                        help="Divide the rate by this (1 = no-op).")
        cumulative_avg = st.checkbox("cumulative_avg", value=True, key="wk_cavg",
                                     help="Kinetics computed with cumulative averaging "
                                          "(relevant for assign.h5 state populations).")

        if statepop == "assign" and assign is None and source != "Server path":
            st.warning("statepop='assign' needs an assign.h5 file.")

        section("RED correction")
        red = st.checkbox("Apply RED scheme", value=False, key="wk_red",
                          help="Rate from Event Durations correction (from the "
                               "`durations` dataset).")
        rtp = st.number_input("red_timepoints (0=auto)", min_value=0, value=0, step=1,
                              key="wk_rtp",
                              help="pcoord frames per iteration for RED resolution; "
                                   "0 auto-detects from assign.h5 npts.")
        red_timepoints = None if int(rtp) == 0 else int(rtp)

        section("Reference")
        ref_txt = st.text_input("ref value(s)", value="", key="wk_ref",
                                help="Experimental rate(s) as horizontal dashed line(s).")
        ref_values = parse_floats(ref_txt)

        # ---- formatting --------------------------------------------------- #
        with st.expander("Plot formatting", expanded=False):
            c1, c2 = st.columns(2)
            style = c1.text_input("style", value="default", key="wk_style")
            color = c2.text_input("color", value="", key="wk_color")
            c1, c2 = st.columns(2)
            label = c1.text_input("label", value="", key="wk_label")
            lw = c2.text_input("linewidth", value="", key="wk_lw")
            linewidth = float(lw) if lw.strip() else None
            linestyle = st.text_input("linestyle", value="-", key="wk_ls") or "-"

            c1, c2 = st.columns(2)
            xlabel = c1.text_input("xlabel", value="", key="wk_xlabel")
            ylabel = c2.text_input("ylabel", value="", key="wk_ylabel")
            c1, c2 = st.columns(2)
            xlim = parse_lim(c1.text_input("xlim (lo,hi)", value="", key="wk_xlim"))
            ylim = parse_lim(c2.text_input("ylim (lo,hi)", value="", key="wk_ylim"))
            c1, c2 = st.columns(2)
            title = c1.text_input("title", value="", key="wk_title")
            suptitle = c2.text_input("suptitle", value="", key="wk_suptitle")
            c1, c2 = st.columns(2)
            axvline = parse_floats(c1.text_input("axvline(s)", value="", key="wk_axvline"))
            axhline = parse_floats(c2.text_input("axhline(s)", value="", key="wk_axhline"))
            grid = st.checkbox("grid", value=False, key="wk_grid")

        pp_mode, pp_path, pp_code, pp_func = postprocess_controls(
            "wk", default_inline="ax.axhline(1e3, color='k', ls='--')")

    # ---- assemble kwargs -------------------------------------------------- #
    kin_kwargs = dict(
        direct=direct, assign=assign,
        tau=float(tau_ps) * 1e-12, state=int(state),
        statepop=statepop, flux_units=flux_units, x_units=x_units,
        concentration=float(concentration), cumulative_avg=bool(cumulative_avg),
        red=bool(red), red_timepoints=red_timepoints,
        color=color.strip() or None, label=label.strip() or None,
        linewidth=linewidth, linestyle=linestyle,
    )
    # formatting options are read by Kinetics._unpack_plot_options via self.kwargs
    fmt_kwargs = {}
    if xlabel.strip():
        fmt_kwargs["xlabel"] = xlabel
    if ylabel.strip():
        fmt_kwargs["ylabel"] = ylabel
    if xlim:
        fmt_kwargs["xlim"] = xlim
    if ylim:
        fmt_kwargs["ylim"] = ylim
    if title.strip():
        fmt_kwargs["title"] = title
    if suptitle.strip():
        fmt_kwargs["suptitle"] = suptitle
    if grid:
        fmt_kwargs["grid"] = True
    if axvline:
        fmt_kwargs["axvline"] = axvline
    if axhline:
        fmt_kwargs["axhline"] = axhline

    apply_style_safe(style, "wekap")

    # ---- render in the sticky plot column: warnings/loading/plot together -- #
    with plot_col, st.container(key="plot_sticky_wk"):
        notice = st.container()  # pinned above the plot with it
        try:
            with st.spinner("Computing rates..."):
                k = wekap.Kinetics(**kin_kwargs, **fmt_kwargs)
                if multi_direct:
                    k.plot_multi_rates(multi_direct)
                else:
                    k.plot_rate()
                k._unpack_plot_options()
                if ref_values:
                    for rv in ref_values:
                        k.plot_ref_vals(rv)
            if pp_mode != "none":
                try:
                    run_postprocess(k, "file" if pp_mode == "file path" else "inline",
                                    pp_path, pp_code, pp_func, extra_ns={"wekap": wekap, "k": k})
                except Exception as exc:
                    notice.warning(f"Postprocessing failed: {exc}")
            fig = k.fig
            fig.tight_layout()
        except Exception as exc:
            notice.error(f"Plotting failed: {exc}")
            return

        st.pyplot(fig, use_container_width=True)
        figure_download_button(fig, "wk", "wekap_plot.png")
    plt.close(fig)

    # ---- equivalent Python ------------------------------------------------ #
    echo_kwargs = {k_: v for k_, v in {**kin_kwargs, **fmt_kwargs}.items() if v is not None}
    call = "k.plot_multi_rates([...])" if multi_direct else "k.plot_rate()"
    with st.expander("Equivalent Python", expanded=False):
        st.code(
            "import wekap\n\n"
            "k = wekap.Kinetics(\n"
            + "".join(f"    {key}={val!r},\n" for key, val in echo_kwargs.items())
            + ")\n"
            + call,
            language="python",
        )
