"""Analytical amplitudes vs fits to analytical (dashed) and RWZ (dotted) data.

Columns: co-/counter-rotating for DATASET, plus co-rotating radial infall (RADIAL_DATASET).
"""

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator

from core import (load_package, load_rwz, load_analytical_model,
                  get_analytical_amplitude, mode_set, package_path)
from fitting import fit_amplitudes
from plotting import mark_times, config, overtone_stack, save

# ── Configuration ──────────────────────────────────────────────────────────────

DATASET        = "ID00"   # main package (co-/counter-rotating columns)
RADIAL_DATASET = "ID10"   # radial-infall package (third, co-rotating column)
DATA_TYPE_QC   = "strain" # "strain" or "psi4" — main (QC) columns
DATA_TYPE_RAD  = "psi4"   # "strain" or "psi4" — radial column
LAMBDA_REG     = 1e-4      # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.01)
T_FIT         = 100.0
INCLUDE_RETRO = True

X_LIM      = (T0_ARRAY[0], T0_ARRAY[-1])   # x-limits for the top block (n=0..3)
X_LIM_HIGH = (0, 22.0)                      # narrower x-limits for the bottom block (n=4..7)
Y_LOG_PAD  = 0.4   # fractional log-decade padding added above/below each row's data range
LABEL_Y_FRAC = 0.20 # top fraction of each panel kept clear for the n= label
LABEL_X_FRAC = 0.33 # left fraction of the x-range the n= label spans

MODES = mode_set(N_MAX, INCLUDE_RETRO)

def _row_ylim(curves, xlim, apply_pad=True):
    """Log (ymin, ymax) of the curves within ``xlim``, raised so the top-left n label clears the data."""
    vis = []
    for x, y in curves:
        x = np.asarray(x)
        y = np.asarray(y)
        m = (x >= xlim[0]) & (x <= xlim[1]) & (y > 0)
        if np.any(m):
            vis.append((x[m], y[m]))
    ys = np.concatenate([y for _, y in vis])
    lo, hi = np.log10(ys.min()), np.log10(ys.max())
    pad = Y_LOG_PAD * max(hi - lo, 1.0) if apply_pad else 0.0
    lo -= pad
    hi += pad

    x_left = xlim[0] + LABEL_X_FRAC * (xlim[1] - xlim[0])
    left_max = -np.inf
    for x, y in vis:
        m = x <= x_left
        if np.any(m):
            left_max = max(left_max, np.log10(y[m]).max())
    if np.isfinite(left_max):
        hi = max(hi, lo + (left_max - lo) / (1.0 - LABEL_Y_FRAC))
    return 10 ** lo, 10 ** hi


def _panel(ax, t_td, A_row, t0_array, C_ana, C_rwz, color, meta, n, xlim):
    """Analytical line and both fits; time markers only if ``meta`` is given."""
    ax.semilogy(t_td, A_row, color=color, lw=0.8, alpha=0.5, zorder=1)
    if C_ana is not None:
        ax.semilogy(t0_array, C_ana, color=color, lw=0.8, ls="--", zorder=3)
        ax.semilogy(t0_array, C_rwz, color=color, lw=0.8, ls=":",  zorder=3)
    if meta is not None:
        mark_times(ax, meta, lw=0.5)
    ax.set_xlim(*xlim)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
    ax.text(0.04, 0.95, rf"$n={n}$", transform=ax.transAxes, va="top", ha="left")


def plot_amplitudes(t0_array, C_norm_ana, C_norm_rwz, t_td, A_td,
                    C_norm_ana_rad, C_norm_rwz_rad, t_td_rad, A_td_rad, meta, modes):
    """One panel per mode: main co-/counter-rotating columns, then the radial column on its own y-scale."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}

    # 4 grid columns: main co-, main counter-, thin spacer, radial co-.
    fig, axes, grid_row, mid = overtone_stack(N_MAX, ncols=4, width_ratios=[1, 1, 0.2, 1],
                                              left=0.13, bottom=0.10)
    for ax in axes[:, 2]:
        ax.set_visible(False)

    for n in range(N_MAX + 1):
        color = config.colors[n]
        xlim = X_LIM if n < mid else X_LIM_HIGH
        mask_qc  = (t_td     >= xlim[0]) & (t_td     <= xlim[1])
        mask_rad = (t_td_rad >= xlim[0]) & (t_td_rad <= xlim[1])

        # ── Main package: co-rotating (col 0) and counter-rotating (col 1) ──
        qc_curves = []
        for sign in (1, -1):
            td_col  = 0 if sign == 1 else 1
            col_idx = mode_col.get((sign, n))
            qc_curves.append((t_td[mask_qc], np.abs(A_td[td_col, n])[mask_qc]))
            if col_idx is not None:
                qc_curves.append((t0_array, C_norm_ana[:, col_idx]))
                qc_curves.append((t0_array, C_norm_rwz[:, col_idx]))
        qc_ylim = _row_ylim(qc_curves, xlim, apply_pad=n in (0, 1))

        gr = grid_row(n)
        for qc_col, sign in enumerate((1, -1)):
            ax      = axes[gr, qc_col]
            td_col  = 0 if sign == 1 else 1
            col_idx = mode_col.get((sign, n))
            _panel(ax, t_td[mask_qc], np.abs(A_td[td_col, n])[mask_qc], t0_array,
                   None if col_idx is None else C_norm_ana[:, col_idx],
                   None if col_idx is None else C_norm_rwz[:, col_idx],
                   color, meta, n, xlim)
            ax.set_ylim(*qc_ylim)
            if qc_col == 0:
                ax.set_ylabel(r"$|\hat{C}|$")
            else:
                ax.tick_params(labelleft=False)
            if n not in (mid - 1, N_MAX):
                ax.tick_params(labelbottom=False)

        # ── Radial package: co-rotating only (col 3), independent y-scale ───
        ax      = axes[gr, 3]
        col_idx = mode_col.get((1, n))
        rad_curves = [(t_td_rad[mask_rad], np.abs(A_td_rad[0, n])[mask_rad])]
        if col_idx is not None:
            rad_curves.append((t0_array, C_norm_ana_rad[:, col_idx]))
            rad_curves.append((t0_array, C_norm_rwz_rad[:, col_idx]))
        _panel(ax, t_td_rad[mask_rad], np.abs(A_td_rad[0, n])[mask_rad], t0_array,
               None if col_idx is None else C_norm_ana_rad[:, col_idx],
               None if col_idx is None else C_norm_rwz_rad[:, col_idx],
               color, None, n, xlim)
        ax.set_ylim(*_row_ylim(rad_curves, xlim, apply_pad=n in (0, 1)))
        if n not in (mid - 1, N_MAX):
            ax.tick_params(labelbottom=False)

    axes[0, 0].set_title("QC co-rotating")
    axes[0, 1].set_title("QC counter-rotating")
    axes[0, 3].set_title("Radial co-rotating")

    for col in (0, 1, 3):
        axes[grid_row(N_MAX), col].set_xlabel(r"$t_0$ [$M$]")

    fig.legend(handles=[
        Line2D([0], [0], color="k", alpha=0.5, label=r"$|\hat{A}(t)|$"),
        Line2D([0], [0], color="k", ls="--", label=r"$|\hat{C}(t_0)|$ (analytical)"),
        Line2D([0], [0], color="k", ls=":",  label=r"$|\hat{C}(t_0)|$ (perturbative)"),
    ], loc="lower center", ncol=3, fontsize=7, frameon=False, bbox_to_anchor=(0.55, 0.0))

    return fig

def fit_package(dataset, data_type):
    """(t_td, A_td, C_norm_ana, C_norm_rwz, meta) for one package."""
    package_dir = package_path(dataset)
    meta, amp_data = load_package(package_dir, data_type=data_type)
    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
    fits = []
    for source, (t, data) in (
            ("analytical", load_analytical_model(package_dir, data_type=data_type,
                                                 meta=meta, data=amp_data)),
            ("RWZ", load_rwz(package_dir, data_type=data_type, meta=meta))):
        print(f"[{dataset}] Fitting {source} data...", flush=True)
        fits.append(fit_amplitudes(t, data, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG))
    return t_td, A_td, *fits, meta


def main():
    t_td, A_td, C_ana, C_rwz, meta = fit_package(DATASET, DATA_TYPE_QC)
    t_td_rad, A_td_rad, C_ana_rad, C_rwz_rad, _ = fit_package(RADIAL_DATASET, DATA_TYPE_RAD)

    fig = plot_amplitudes(T0_ARRAY, C_ana, C_rwz, t_td, A_td,
                          C_ana_rad, C_rwz_rad, t_td_rad, A_td_rad, meta, MODES)
    save(fig, "14_amplitudes_comp")


if __name__ == "__main__":
    main()
