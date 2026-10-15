"""Mismatch vs t0 against RWZ data for co-rotating overtones n <= N.

Solid: analytical amplitudes frozen at A(t0); dotted: least-squares fit.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import (load_package, load_rwz, analytical_constant_amplitude_mismatch_sweep,
                  mode_set, package_path)
from fitting import ls_mismatch_sweep
from plotting import config, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
LAMBDA_REG    = 0.0        # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(0.0, 40.0, 0.1)
T_FIT         = 100.0

PACKAGE_DIR  = package_path(DATASET)
N_MAX_VALUES = list(range(N_MAX + 1))


def plot_mismatch(t0_array, qnm_mm, analytical_mm, meta):
    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.14)

    for i, n_max in enumerate(N_MAX_VALUES):
        color = config.colors[n_max]
        ax.semilogy(t0_array, qnm_mm[i],        color=color, ls=":")
        ax.semilogy(t0_array, analytical_mm[i], color=color, ls="-")
    mark_times(ax, meta)

    ax.add_artist(ax.legend(handles=[
        Line2D([0], [0], color=config.colors[n_max], label=rf"$N={n_max}$")
        for n_max in N_MAX_VALUES
    ], fontsize=7, ncol=3, loc="upper right"))
    ax.legend(handles=[
        Line2D([0], [0], color="k", label="Analytical"),
        Line2D([0], [0], color="k", ls=":", label="Least-squares"),
    ], loc="upper right", bbox_to_anchor=(1.0, 0.72), bbox_transform=ax.transAxes,
       fontsize=7)

    ax.set_xlabel(r"$t_0$ [$M$]")
    ax.set_ylabel("Mismatch")
    ax.set_xlim(t0_array[0], t0_array[-1])
    ax.set_ylim(1e-7, 10)
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t, data = load_rwz(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta)

    qnm_mm, analytical_mm = [], []
    for n_max in N_MAX_VALUES:
        print(f"N={n_max} ...", flush=True)
        modes = mode_set(n_max, include_retro=False)
        qnm_mm.append(ls_mismatch_sweep(t, data, T0_ARRAY, modes, T_FIT, lambda_reg=LAMBDA_REG))
        analytical_mm.append(analytical_constant_amplitude_mismatch_sweep(
            t, data, T0_ARRAY, amp_data, meta, modes, T_FIT))

    save(plot_mismatch(T0_ARRAY, qnm_mm, analytical_mm, meta), "4_mismatch_pert")


if __name__ == "__main__":
    main()
