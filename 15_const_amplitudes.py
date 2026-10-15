"""Analytical TD amplitudes and fits vs the table's constant amplitudes (scaled by core.constant_amplitude_scale)."""

import numpy as np
from matplotlib.lines import Line2D

from core import (load_package, load_waveform, get_analytical_amplitude,
                  load_constant_amplitudes, peak_frame_times, mode_set, package_path)
from fitting import fit_amplitudes
from plotting import config, amplitude_grid, fit_linestyle, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # the constant-amplitude table is for ID00 only
DATA_TYPE     = "strain"   # the table holds strain amplitudes
USE_RWZ       = False
LAMBDA_REG    = 1e-4       # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.01)
T_FIT         = 100.0
INCLUDE_RETRO = True

SHOW_REDSHIFT = False      # also draw |A_const + A_red(t)|

# TD lines thick and faded, constant levels thin.
TD_LW, TD_ALPHA = 1.1, 0.5
CONST_LW        = 0.6
RED_LS, RED_LW  = "-.", 0.7
LEGEND_FS       = 8

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_LIM = (1e-3, 1e8)

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)
FIT_LS      = fit_linestyle(USE_RWZ)


def plot_amplitudes(t0_array, C_norm, t_td, A_td, A_const, A_red, meta, modes):
    """TD, fitted and constant amplitudes on the 2x2 grid; ``A_red`` optionally adds the redshift term."""
    t_bounce = peak_frame_times(meta)["t_bounce"]
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}
    fig, axes, n_groups = amplitude_grid(N_MAX)

    for row, sign in enumerate((1, -1)):
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            x_lim = X_LIM if col == 0 else X_LIM_HIGH
            for n in n_group:
                color = config.colors[n]
                ax.semilogy(t_td, np.abs(A_td[row, n]), color=color, lw=TD_LW, alpha=TD_ALPHA,
                            label=rf"$n={n}$")
                if (sign, n) in mode_col:
                    ax.semilogy(t0_array, C_norm[:, mode_col[(sign, n)]], color=color, ls=FIT_LS)
                ax.semilogy([t_bounce, x_lim[1]], [np.abs(A_const[row, n])] * 2,
                            color=color, lw=CONST_LW)
                if A_red is not None:
                    ax.semilogy(t_td, np.abs(A_const[row, n] + A_red[row, n]),
                                color=color, ls=RED_LS, lw=RED_LW)
            mark_times(ax, meta)
            ax.set_xlim(*x_lim)
            ax.set_ylim(*Y_LIM)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, loc="lower right" if col == 1 else "upper left",
                                        frameon=False))

    handles = [
        Line2D([0], [0], color="k", lw=TD_LW, alpha=TD_ALPHA, label=r"$|\hat{A}(t)|$"),
        Line2D([0], [0], color="k", ls=FIT_LS, label=r"$|\hat{C}(t_0)|$"),
        Line2D([0], [0], color="k", lw=CONST_LW, label=r"$|\hat{A}_{\rm const}|$"),
    ]
    if A_red is not None:
        handles.append(Line2D([0], [0], color="k", ls=RED_LS, lw=RED_LW,
                              label=r"$|\hat{A}_{\rm const}+\hat{A}_{\rm red}|$"))
    axes[1, 1].legend(handles=handles, loc="lower right", fontsize=LEGEND_FS)
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t, data = load_waveform(PACKAGE_DIR, USE_RWZ, DATA_TYPE, meta=meta, data=amp_data)

    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)
    t_red, A_red = load_constant_amplitudes("redshift", n_max=N_MAX, data=amp_data, meta=meta)
    assert np.array_equal(t_red, t_td)

    # Check of the conversion: late on, A_td is the table's constant + redshift.
    late = t_td > 20.0
    model = A_const[0, 0] + A_red[0, 0, late]
    err = np.max(np.abs(A_td[0, 0, late] - model) / np.abs(model))
    print(f"220: max |A_td - (const + redshift)| / |const + redshift| for t > 20 M = {err:.2e}")

    C_norm = fit_amplitudes(t, data, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)

    fig = plot_amplitudes(T0_ARRAY, C_norm, t_td, A_td, A_const,
                          A_red if SHOW_REDSHIFT else None, meta, MODES)
    save(fig, "15_const_amplitudes" + ("_withred" if SHOW_REDSHIFT else ""))


if __name__ == "__main__":
    main()
