"""Fits to constant-amplitude data without decay correction: raw |C(t0)| against |A_const| exp(-gamma t0)."""

import numpy as np
from matplotlib.lines import Line2D

from core import (load_package, load_analytical_model, load_constant_amplitudes, peak_frame_times,
                  qnm_frequencies, mode_set, package_path)
from fitting import run_amplitude_sweep
from plotting import config, amplitude_grid, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET    = "ID00"     # the constant-amplitude table is for ID00 only
LAMBDA_REG = 0.0
N_MAX      = 7
T_FIT      = 100.0      # truncated by the end of the data (~57 M)
T0_END     = 50.0
X_LIM      = (0.0, T0_END)
Y_LIM      = (1e-20, 1e3)

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX)


def main():
    meta, amp_data = load_package(PACKAGE_DIR)
    t, h = load_analytical_model(PACKAGE_DIR, meta=meta, data=amp_data, amplitudes="constant")
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    t0_array = np.arange(t_bounce, T0_END, 0.05)

    C = np.abs(run_amplitude_sweep(t, h, MODES, t0_array, T_FIT, lambda_reg=LAMBDA_REG)[0])
    gamma = -qnm_frequencies(MODES, 0.0, 1.0).imag
    col = {(sign, n): i for i, (_, _, n, sign) in enumerate(MODES)}

    fig, axes, n_groups = amplitude_grid(N_MAX)
    for row, sign in enumerate((1, -1)):
        for c, n_group in enumerate(n_groups):
            ax = axes[row, c]
            for n in n_group:
                i = col[(sign, n)]
                ax.semilogy(t0_array, np.abs(A_const[row, n]) * np.exp(-gamma[i] * t0_array),
                            color=config.colors[n], label=rf"$n={n}$")
                ax.semilogy(t0_array, C[:, i], color=config.colors[n], ls="--")
            mark_times(ax, meta)
            ax.set_xlim(*X_LIM)
            ax.set_ylim(*Y_LIM)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, loc="upper right", frameon=False))

    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="k", label=r"$|\hat{A}_{\rm const}|\,e^{-\gamma t_0}$"),
        Line2D([0], [0], color="k", ls="--", label=r"$|C(t_0)|$"),
    ], loc="lower left", fontsize=7)
    save(fig, "17_const_fit_raw")


if __name__ == "__main__":
    main()
