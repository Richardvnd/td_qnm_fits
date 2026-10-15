"""Mismatch vs t0 for co-rotating overtones n <= N against analytical data with the same modes.

Solid: analytical amplitudes frozen at A(t0); dashed: least-squares fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import (load_package, load_rwz, get_analytical_model,
                  analytical_constant_amplitude_mismatch_sweep, mode_set, package_path)
from fitting import ls_mismatch_sweep
from plotting import config, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
USE_RWZ       = False      # True → full RWZ data; False → mode-limited analytical reconstruction
LAMBDA_REG    = 0.0        # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(0.0, 40.0, 0.1)
T_FIT         = 100.0

PACKAGE_DIR = package_path(DATASET)


def plot_mismatch(t0_array, qnm_mm, analytical_mm, meta):
    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.14)

    for n in range(N_MAX + 1):
        color = config.colors[n]
        ax.semilogy(t0_array, qnm_mm[n],        color=color, ls="--")
        ax.semilogy(t0_array, analytical_mm[n], color=color, ls="-")
    mark_times(ax, meta)

    ax.add_artist(ax.legend(handles=[
        Line2D([0], [0], color=config.colors[n], label=rf"$N={n}$")
        for n in range(N_MAX + 1)
    ], fontsize=7, ncol=3, loc="upper right"))

    # Legend starts at t0 = 6M, clear of the time markers.
    x_frac = (6.0 - t0_array[0]) / (t0_array[-1] - t0_array[0])
    ax.legend(handles=[
        Line2D([0], [0], color="k", ls="--", label="Least-squares fit"),
        Line2D([0], [0], color="k", ls="-",  label="Analytical"),
    ], loc="lower left", bbox_to_anchor=(x_frac, 0.02), bbox_transform=ax.transAxes,
       fontsize=7)

    ax.set_xlabel(r"$t_0$ [$M$]")
    ax.set_ylabel("Mismatch")
    ax.set_xlim(t0_array[0], t0_array[-1])
    ax.set_ylim(top=60)
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    if USE_RWZ:
        t, data = load_rwz(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta)

    qnm_mm, analytical_mm = [], []
    for n in range(N_MAX + 1):
        print(f"N={n} ...", flush=True)
        modes = mode_set(n, include_retro=False)
        if not USE_RWZ:
            t, data = get_analytical_model(amp_data, meta, n_max=n, signs=[1])
        qnm_mm.append(ls_mismatch_sweep(t, data, T0_ARRAY, modes, T_FIT, lambda_reg=LAMBDA_REG))
        analytical_mm.append(analytical_constant_amplitude_mismatch_sweep(
            t, data, T0_ARRAY, amp_data, meta, modes, T_FIT))

    save(plot_mismatch(T0_ARRAY, qnm_mm, analytical_mm, meta), "5_mismatch_analytical")


if __name__ == "__main__":
    main()
