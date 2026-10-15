"""Least-squares amplitude sweep (dashed; dotted for RWZ) vs analytical TD amplitudes (solid)."""

import numpy as np
from matplotlib.lines import Line2D

from core import (load_package, load_waveform, get_analytical_amplitude, load_constant_amplitudes,
                  mode_set, package_path, peak_frame_times)
from fitting import fit_amplitudes
from plotting import config, amplitude_grid, log_ylim, fit_linestyle, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
USE_RWZ       = False      # True → RWZ data; False → analytical reconstruction
LAMBDA_REG    = 1e-4       # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.01)
T_FIT         = 100.0
INCLUDE_RETRO = True

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_WINDOW = 30.0   # y floor: lowest curve between the bounce and this t0 [M]; ceiling: highest constant
Y_PAD = 0.1
LW = 0.65                          # data lines
CONST_COLOR, CONST_ALPHA, CONST_LW = "r", 0.5, 1.2   # table constants, across the full x range

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)
FIT_LS      = fit_linestyle(USE_RWZ)


def plot_amplitudes(t0_array, C_norm, t_td, A_td, A_const, meta, modes):
    """Analytical (solid) vs least-squares amplitudes on the 2x2 grid; table constants in red."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}
    fig, axes, n_groups = amplitude_grid(N_MAX)
    t_bounce = peak_frame_times(meta)["t_bounce"]

    for row, sign in enumerate((1, -1)):
        in_view = []
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            for n in n_group:
                color = config.colors[n]
                ax.semilogy(t_td, np.abs(A_td[row, n]), color=color, lw=LW, label=rf"$n={n}$")
                if (sign, n) in mode_col:
                    ax.semilogy(t0_array, C_norm[:, mode_col[(sign, n)]], color=color, lw=LW, ls=FIT_LS)
            xlim = X_LIM if col == 0 else X_LIM_HIGH
            t_hi = min(Y_WINDOW, xlim[1])
            in_view += [np.abs(A_td[row, n_group])[:, (t_td >= t_bounce) & (t_td <= t_hi)].min(),
                        C_norm[(t0_array >= t_bounce) & (t0_array <= t_hi)][:, [mode_col[(sign, n)] for n in n_group]].min(),
                        np.abs(A_const[row, n_group])]
            ax.hlines(np.abs(A_const[row, n_group]), *xlim, colors=CONST_COLOR,
                      alpha=CONST_ALPHA, lw=CONST_LW, zorder=0)
            mark_times(ax, meta)
            ax.set_xlim(*xlim)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, loc="lower right" if col == 1 else "upper left",
                                        frameon=False))
        axes[row, 0].set_ylim(*log_ylim(in_view, Y_PAD))

    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="k", lw=LW, label=r"$|\hat{A}(t)|$"),
        Line2D([0], [0], color="k", lw=LW, ls=FIT_LS, label=r"$|\hat{C}(t_0)|$"),
        Line2D([0], [0], color=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW, label=r"$|\hat{A}_{\rm const}|$"),
    ], loc="lower right")
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t, data = load_waveform(PACKAGE_DIR, USE_RWZ, DATA_TYPE, meta=meta, data=amp_data)

    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
    C_norm = fit_amplitudes(t, data, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)

    save(plot_amplitudes(T0_ARRAY, C_norm, t_td, A_td, A_const, meta, MODES), "1_amplitudes")


if __name__ == "__main__":
    main()
