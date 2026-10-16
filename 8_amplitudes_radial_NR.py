"""Radial infall: analytical amplitudes vs fits to RWZ (dotted) and SXS:BBH:3991 (dash-dot).

RWZ and NR fits set (2,2,0,+) = 1 at NORM_T0; the analytical curve shares the RWZ factor.
Arrows: late-time constant amplitudes from the ID10 table (n <= 6), as psi4 (-omega^2 A), with the RWZ factor.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import (load_package, load_rwz, get_analytical_amplitude, load_constant_amplitudes, load_sxs_data,
                  mode_set, package_path)
from fitting import fit_amplitudes
from plotting import config, mark_times, mark_constants, constant_marker_handle, overtone_groups, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID10"     # radial infall (e=1)
DATA_TYPE     = "psi4"
SXS_ID        = "SXS:BBH:3991"
LAMBDA_REG    = 1e-4
N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.1)
T_FIT         = 100.0
INCLUDE_RETRO = True

NORM_T0 = 10.0   # [M] normalise (2,2,0,+) amplitude to 1 at this t0

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_LIM = (1e-4, 1e8)

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)


def plot_amplitudes(t0_array, C_nr, C_rwz, t_td, A_td, A_const, meta, modes):
    """One row (co-rotating), low/high overtone columns; light-ring and bounce times marked."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}

    fig, axes = plt.subplots(1, 2, figsize=(config.fig_width_2, config.fig_height * 1.1),
                             sharey="row", gridspec_kw={"wspace": 0.04})
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.90, bottom=0.16)

    for col, n_group in enumerate(overtone_groups(N_MAX)):
        ax = axes[col]
        for n in n_group:
            color = config.colors[n]
            idx = mode_col[(1, n)]
            ax.semilogy(t_td, np.abs(A_td[0, n]), color=color, label=rf"$n={n}$")
            ax.semilogy(t0_array, C_nr[:, idx], color=color, ls="-.")
            ax.semilogy(t0_array, C_rwz[:, idx], color=color, ls=":")
        mark_constants(ax, A_const[0, n_group], [config.colors[n] for n in n_group])
        mark_times(ax, meta)
        ax.set_xlim(*(X_LIM if col == 0 else X_LIM_HIGH))
        ax.set_ylim(*Y_LIM)
        ax.set_xlabel(r"$t_0$ [$M$]")
        ax.set_title(rf"$n = {n_group[0]}$--${n_group[-1]}$")
        ax.add_artist(ax.legend(fontsize=7, loc="lower right" if col == 1 else "upper left",
                                frameon=False))
    axes[0].set_ylabel("Amplitude")
    axes[1].tick_params(labelleft=False)

    axes[0].legend(handles=[
        Line2D([0], [0], color="k", label=r"$|\hat{A}(t)|$"),
        Line2D([0], [0], color="k", ls=":", label=r"$|\hat{C}(t_0)|$ (Pert)"),
        Line2D([0], [0], color="k", ls="-.", label=r"$|\hat{C}(t_0)|$ (NR)"),
        constant_marker_handle(),
    ], loc="lower right", ncol=2, fontsize=6.5, columnspacing=0.6, handlelength=1.4,
       handletextpad=0.3, borderaxespad=0.3)
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta, data_type=DATA_TYPE)
    t_rwz, data_rwz = load_rwz(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta)

    print(f"Loading {SXS_ID} ({DATA_TYPE})...", flush=True)
    t_nr, h22_nr, Mf, chif = load_sxs_data(SXS_ID, data_type=DATA_TYPE)
    print(f"  Mf = {Mf:.6f}  chif = {chif:.6f}", flush=True)

    print("Fitting NR and RWZ data...", flush=True)
    C_nr = fit_amplitudes(t_nr, h22_nr, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG,
                          Mf=Mf, chif=chif)
    C_rwz = fit_amplitudes(t_rwz, data_rwz, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)

    ref_col = MODES.index((2, 2, 0, 1))
    ref_idx = np.argmin(np.abs(T0_ARRAY - NORM_T0))
    rwz_ref = C_rwz[ref_idx, ref_col]
    A_td    = A_td / rwz_ref
    A_const = A_const / rwz_ref
    C_rwz   = C_rwz / rwz_ref
    C_nr    = C_nr / C_nr[ref_idx, ref_col]

    save(plot_amplitudes(T0_ARRAY, C_nr, C_rwz, t_td, A_td, A_const, meta, MODES), "8_amplitudes_radial_NR")


if __name__ == "__main__":
    main()
