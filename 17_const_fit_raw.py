"""Unregularised fits to constant-amplitude data (Fig. 19).

DECAY_CORRECT = True: decay-corrected |C(t0)| against the flat |A_const|.
DECAY_CORRECT = False: raw |C(t0)| against |A_const| exp(-gamma t0), which shows the fit floor directly.
"""

import numpy as np
from matplotlib.lines import Line2D

from core import (load_package, load_analytical_model, load_constant_amplitudes, peak_frame_times,
                  qnm_frequencies, mode_set, package_path)
from fitting import run_amplitude_sweep, normalize_amplitudes
from plotting import config, amplitude_grid, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # the constant-amplitude table is for ID00 only
DECAY_CORRECT = True
LAMBDA_REG    = 0.0
N_MAX         = 7
T_FIT         = 100.0      # truncated by the end of the data (~57 M)

# Fits start at the bounce: before it the constant-amplitude modes grow exponentially backwards.
if DECAY_CORRECT:
    X_LIM      = (-10.0, 40.0)   # as Fig. 3
    X_LIM_HIGH = (0, 20)
    Y_LIM      = (1e-3, 1e8)
else:
    X_LIM      = (0.0, 50.0)
    X_LIM_HIGH = X_LIM
    Y_LIM      = (1e-20, 1e3)

CONST_LW = 1.4             # bold expected values
FIT_LW   = 0.8
NAME     = "17_const_fit" if DECAY_CORRECT else "17_const_fit_raw"

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX)


def main():
    meta, amp_data = load_package(PACKAGE_DIR)
    t, h = load_analytical_model(PACKAGE_DIR, meta=meta, data=amp_data, amplitudes="constant")
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    t0_array = np.arange(t_bounce, X_LIM[1], 0.05)

    C = run_amplitude_sweep(t, h, MODES, t0_array, T_FIT, lambda_reg=LAMBDA_REG)[0]
    if DECAY_CORRECT:
        C = normalize_amplitudes(C, MODES, t0_array, chif=0.0, Mf=1.0)
    else:
        C = np.abs(C)
    gamma = -qnm_frequencies(MODES, 0.0, 1.0).imag
    col = {(sign, n): i for i, (_, _, n, sign) in enumerate(MODES)}

    fig, axes, n_groups = amplitude_grid(N_MAX)
    for row, sign in enumerate((1, -1)):
        for c, n_group in enumerate(n_groups):
            ax = axes[row, c]
            x_lim = X_LIM if c == 0 else X_LIM_HIGH
            for n in n_group:
                i = col[(sign, n)]
                color = config.colors[n]
                if DECAY_CORRECT:
                    ax.semilogy([t_bounce, x_lim[1]], [np.abs(A_const[row, n])] * 2,
                                color=color, lw=CONST_LW, label=rf"$n={n}$")
                else:
                    ax.semilogy(t0_array, np.abs(A_const[row, n]) * np.exp(-gamma[i] * t0_array),
                                color=color, lw=CONST_LW, label=rf"$n={n}$")
                ax.semilogy(t0_array, C[:, i], color="k", ls="--", lw=FIT_LW, zorder=4)
            mark_times(ax, meta)
            ax.set_xlim(*x_lim)
            ax.set_ylim(*Y_LIM)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, frameon=False,
                                        loc="lower right" if DECAY_CORRECT and c == 1 else "upper right"))

    const_label = (r"$|\hat{A}_{\rm const}|$" if DECAY_CORRECT
                   else r"$|\hat{A}_{\rm const}|\,e^{-\gamma t_0}$")
    fit_label = r"$|\hat{C}(t_0)|$" if DECAY_CORRECT else r"$|C(t_0)|$"
    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="k", lw=CONST_LW, label=const_label),
        Line2D([0], [0], color="k", ls="--", lw=FIT_LW, label=fit_label),
    ], loc="lower right" if DECAY_CORRECT else "lower left", fontsize=7)
    save(fig, NAME)


if __name__ == "__main__":
    main()
