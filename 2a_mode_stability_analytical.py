"""Fig. 2 fitted to the analytical reconstruction (no tail, no numerical noise) instead of RWZ.

Top: |A(t)|, |C(t0)| and the table constant; bottom: |C(t0) - A(t0)|.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from core import (load_package, load_waveform, get_analytical_amplitude, load_constant_amplitudes,
                  mode_set, package_path)
from fitting import fit_amplitudes
from plotting import config, fit_linestyle, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
USE_RWZ       = False      # fig. 2 uses RWZ (True)
LAMBDA_REG    = 0.0        # ridge regularisation (0.0 = none)

PLOT_MODES    = [(0, 1), (1, 1)]   # [(n, sign), ...]: one column per mode
TOP_YLIM      = {0: (0.3, 10), 1: (None, 500)}   # top-panel ylim per n (None = auto)
RESID_YTICKS  = {0: [1e-4, 1e-2, 1], 1: [1e-3, 1e-1, 10]}   # residual-panel ticks per n
CONST_COLOR, CONST_ALPHA, CONST_LW = "r", 0.5, 1.5  # table constant amplitude

N_MAX         = 7
T0_ARRAY      = np.arange(-30.0, 40.0, 0.5)
T_FIT         = 15.0
INCLUDE_RETRO = True

PACKAGE_DIR = package_path(DATASET)
FIT_LS      = fit_linestyle(USE_RWZ)


def plot_stability(t_td, columns, C_by_nmax, meta):
    """``columns``: (n, sign, |A(t)|, |A_const|) per column; ``C_by_nmax``: N -> {(n, sign): |C(t0)|}."""
    n_cols = len(columns)
    fig, axes = plt.subplots(
        2, n_cols, figsize=(config.fig_width_2, config.fig_height),
        sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.06, "wspace": 0.132},
    )
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.06, right=0.985, top=0.93, bottom=0.10)

    mask_td = (t_td >= T0_ARRAY[0]) & (t_td <= T0_ARRAY[-1])
    for col, (n_plot, sign, A_analytic, A_const) in enumerate(columns):
        ax_top, ax_bot = axes[0, col], axes[1, col]
        A_at_t0 = np.interp(T0_ARRAY, t_td, A_analytic)

        ax_top.semilogy(t_td[mask_td], A_analytic[mask_td], color="k", lw=1.2, zorder=10)
        ax_top.axhline(A_const, color=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW, zorder=11)
        for n_max, C in C_by_nmax.items():
            if (n_plot, sign) in C:
                ax_top.semilogy(T0_ARRAY, C[(n_plot, sign)], color=config.colors[n_max], ls=FIT_LS)
                ax_bot.semilogy(T0_ARRAY, np.abs(C[(n_plot, sign)] - A_at_t0),
                                color=config.colors[n_max])
        ax_top.text(0.03, 0.95, rf"$n={n_plot}$", transform=ax_top.transAxes,
                    ha="left", va="top", fontsize=9, zorder=20,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        ax_bot.set_xlabel(r"$t_0$ [$M$]")
        ax_bot.yaxis.set_major_locator(FixedLocator(RESID_YTICKS[n_plot]))
        if n_plot in TOP_YLIM:
            ax_top.set_ylim(*TOP_YLIM[n_plot])

        for ax in (ax_top, ax_bot):
            mark_times(ax, meta)
            ax.set_xlim(T0_ARRAY[0], T0_ARRAY[-1])

    sign_char = "+" if columns[0][1] == 1 else "-"
    axes[0, 0].set_ylabel("Amplitude")
    axes[1, 0].set_ylabel(rf"$|\hat{{C}}_{{22n{sign_char}}} - \hat{{A}}_{{22n{sign_char}}}|$")
    axes[0, 0].legend(handles=[
        Line2D([0], [0], color=config.colors[n_max], label=rf"$N={n_max}$")
        for n_max in C_by_nmax
    ], fontsize=7, ncol=2, loc="lower right")
    axes[0, -1].legend(handles=[
        Line2D([0], [0], color="k", lw=1.2, label=r"$|\hat{A}(t)|$"),
        Line2D([0], [0], color="k", ls=FIT_LS, label=r"$|\hat{C}(t_0)|$"),
        Line2D([0], [0], color=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW, label=r"$|\hat{A}_{\rm const}|$"),
    ], fontsize=7, loc="lower right")
    fig.align_ylabels(axes[:, 0])
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t, data = load_waveform(PACKAGE_DIR, USE_RWZ, DATA_TYPE, meta=meta, data=amp_data)
    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)

    C_by_nmax = {}
    for n_max in range(N_MAX + 1):
        print(f"n_max={n_max} ...", flush=True)
        modes = mode_set(n_max, INCLUDE_RETRO)
        C_norm = fit_amplitudes(t, data, modes, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)
        C_by_nmax[n_max] = {(n, sign): C_norm[:, i] for i, (_, _, n, sign) in enumerate(modes)}

    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)
    columns = [(n, sign, np.abs(A_td[row, n]), np.abs(A_const[row, n]))
               for n, sign in PLOT_MODES for row in [0 if sign == 1 else 1]]
    save(plot_stability(t_td, columns, C_by_nmax, meta), "2a_mode_stability_analytical")


if __name__ == "__main__":
    main()
