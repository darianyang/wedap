# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0] - 2026-10-07

Fixes h5 files being left open (and locked) by wedap, which could block other programs
such as `w_multi_west` or `w_run` from writing to them, along with fixes to the
`H5save_out`, `succ_only`, `skip_basis` and multiple h5 file options. All changes are
backwards-compatible.

### Added

- **`H5_Pdist.close()` and `with` block support** (also for `H5_Plot`). These are not
  needed for normal use (see Fixed), but a `with` block keeps the h5 file open across
  repeated method calls, e.g. `get_parents()` in a loop, instead of reopening it for
  each call.

### Fixed

- **h5 file left open and locked.** `H5_Pdist`/`H5_Plot` kept the h5 file open for as
  long as the object existed, including after an error (e.g. `last_iter` larger than
  the available iterations), since an interactive session keeps the traceback and
  object alive. HDF5 file locking then blocked other processes from writing to the
  file (`BlockingIOError: [Errno 11] ... unable to lock file`) until the Python session
  was restarted. The file is now opened when needed and closed again after each method
  call, whether it succeeds or raises.
- **`H5save_out` file handling.** The new h5 file was never closed (holding a write
  lock), and the `H5save_out` attribute was replaced by the open file, so a repeated
  `pdist()` call failed. The file is now written under a temporary name and only moved
  into place once complete, so a failed run leaves no partial output and an existing
  output file is untouched. Inputs are checked before copying: an error is raised if
  `H5save_out` is the input file or if an `X/Y/Zsave_name` dataset already exists, and
  a warning is given that only the first file is copied for multiple h5 file input.
- **`succ_only` and `skip_basis` weights with `first_iter`/`step_iter`.** `succ_only`
  raised an `IndexError` with `first_iter > 1`, and both options used misaligned
  weights with `step_iter > 1`. With `H5save_out`, the filtered weights were also
  written to the wrong iterations.
- **`succ_only` and `skip_basis` with multiple h5 files.** The weight filters were
  dropped for all files, giving the same result as without the option. A repeated
  `pdist()` call with `H5save_out` also saved weights from the last file instead of the
  first.

## [1.2.0] - 2026-07-17

Adds the optional Streamlit web app and a related fix to colorbar labeling. All
changes are backwards-compatible.

### Added

- **Optional Streamlit web app** (`pip install "wedap[web]"`, launch with `wedap-web`
  or `streamlit run wedap/web/app.py`). A browser-based front-end covering **all three
  tools** as tabs (each tab keeps its own inputs when switching between them):
  - **wedap** — the `west.h5` workflow: server-path/upload/example input, auto-discovered
    pcoord + aux dataset dropdowns, all plot modes (1D/2D/3D), the full formatting arg
    set, plot tracing, gif making, and postprocessing (file path or inline code).
  - **mdap** — probability distributions/plots from MD analysis data files (`pdist`/`time`
    data types, 1D/2D/3D modes, shared formatting).
  - **wekap** — WE kinetics rates/MFPTs from a `direct.h5` file (tau/state/concentration,
    the RED correction, and multi-replicate averaging with bootstrapped error).

  Each tool provides a PNG download and a copy-pasteable equivalent Python snippet.
  Streamlit is only pulled in via the `web` extra, so the core dependency tree is
  unchanged. See the README for local vs. remote (SSH-tunnel) launch instructions.
- **`H5_Plot.set_cbar_label()`** — a class method that sets the colorbar label from
  the probability units (`self.p_units`).

### Fixed

- **Colorbar label with precomputed arrays.** `wedap.Plot(X, Y, Z, p_units=...)` (and
  the web app) left the colorbar blank because the array-input path skipped
  `H5_Pdist.__init__` and never set `self.p_units`. `H5_Plot` now falls back to the
  `p_units` kwarg so the label (e.g. "Probability", `$-\ln P(x)$`) is applied.

## [1.1.0] - 2026-07-17

Modernizes the tooling and adds several requested features. Most changes are
backwards-compatible; the breaking items are limited to environment/dependency
requirements and the removal of the deprecated GUI.

### Changed (breaking)

- **Removed the `Gooey`-based GUI** and all mentions of it (unmaintained upstream
  since 2021). Use the CLI or Python API instead. Pin to `wedap<1.1.0` if you relied
  on the GUI.
- **Raised the Python and dependency floors.** Minimum Python is now **3.9**. The old
  `numpy<2` / `matplotlib<=3.7.0` upper caps were **dropped** — `wedap` now supports
  **numpy 2.x** and recent matplotlib (tested through numpy 2.5 / matplotlib 3.11 on
  Python 3.12). Pin to `wedap<1.1.0` for older interpreters.

### Added

- **Logging** with `--verbose`/`-v` and `--debug` flags across all three CLIs
  (`wedap`, `mdap`, `wekap`), backed by a shared logger in `wedap/logger.py`. Verbose
  mode echoes a copy-pasteable constructor call (e.g.
  `wekap.Kinetics(direct='...', tau=1e-10, concentration=1.0, red=True, ...)`) so a CLI
  invocation can be lifted directly into a script or notebook.
- **`wekap --red` now computes the RED scheme.** Computes the [RED duration correction](https://www.biorxiv.org/content/10.1101/453647v2.full) from the `durations` dataset directly.
- **`wekap` concentration normalization** via `-c/--concentration` (default `1`), which
  divides the extracted rate by the given molar concentration to yield a
  pseudo-second-order rate.
- **`raw_norm_tot` probability units** (`-pu raw_norm_tot`), which normalizes the
  histogram by the total count so it sums to 1. The existing `raw_norm` (normalize by
  max) is unchanged.
- **Artist kwarg forwarding.** Keyword arguments such as `alpha`, `zorder`, and
  `edgecolor` passed to the plotting classes/CLIs are now threaded through to the
  underlying matplotlib artists (including a new `--alpha` CLI flag). Unsupported kwargs
  are filtered per-artist rather than raising.
- **Friendly array-input API.** New `wedap.Plot`/`wedap.Pdist` and `mdap.Plot`/`mdap.Pdist`
  aliases. Precomputed numpy arrays now feed cleanly into the plotting classes; numpy
  arrays go to `Plot` (e.g. `wedap.Plot(X, Y, Z)`), never `Pdist`. `MD_Plot(X=x, Y=y, Z=z)`
  now accepts precomputed arrays instead of silently recomputing the pdist.
- **`wekap` unit tests** and numpy-array API input tests. CI now runs a numpy-1 and a
  numpy-2 matrix leg across Python 3.10–3.13.

### Fixed

- Fixed a bug where 1D numpy arrays fed to `mdap` errored on the default column index.
- Fixed stale `--style` help text and hard-coded relative data paths in the `wekap`
  demo.

### Removed

- Removed the deprecated `pkg_resources` dependency in favor of
  `importlib.metadata`/`importlib.resources`.
- Removed the top-level `styles/` directory from the packaged distribution (the
  per-package `styles/` copies are retained).

## [1.0.5]
- Untagged pypi release.

## [1.0.4]
- Untagged pypi release.

## [1.0.3]
- Untagged pypi release.

## [1.0.2]
- Untagged pypi release.

## [1.0.1]
- Untagged pypi release.

## [1.0.0]

- Initial tagged release.

[1.3.0]: https://github.com/darianyang/wedap/compare/v1.2.0...v1.3.0
[1.2.0]: https://github.com/darianyang/wedap/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/darianyang/wedap/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/darianyang/wedap/releases/tag/v1.0.0
