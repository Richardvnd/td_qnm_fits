"""Regularised fits to the constant-amplitude analytical data (table A_const, no redshift), alone and overlaid
on the fits to the time-dependent analytical data (as fig. 1), with both sets of analytical amplitudes.

Lines: |A(t)| solid; |A_const| light red; fit to constant data dash-dot; fit to time-dependent data dashed.
The constant-amplitude data start at the bounce, so their fits do too.
"""

import numpy as np
from matplotlib.lines import Line2D

from core import (load_package, load_analytical_model, get_analytical_amplitude, load_constant_amplitudes,
                  mode_set, package_path, peak_frame_times)
from fitting import fit_amplitudes
from plotting import config, amplitude_grid, log_ylim, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # the constant-amplitude table is for ID00 only
LAMBDA_REG    = 1e-4
N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.05)
T_FIT         = 100.0
INCLUDE_RETRO = True

X_LIM      = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_WINDOW   = 30.0          # y range: curves between the bounce and this t0 [M]
Y_PAD      = 0.1
LW         = 0.65
CONST_COLOR, CONST_ALPHA, CONST_LW = "r", 0.5, 1.2
LS_TD, LS_CONST = "--", "-."

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)


def plot(t0_array, t_td, A_td, A_const, C_const, t0_const, meta, C_td=None):
    """2x2 amplitude grid; C_td=None draws the constant-data fit only."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(MODES)}
    fig, axes, n_groups = amplitude_grid(N_MAX)
    t_bounce = peak_frame_times(meta)["t_bounce"]

    for row, sign in enumerate((1, -1)):
        in_view = []
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            xlim = X_LIM if col == 0 else X_LIM_HIGH
            t_hi = min(Y_WINDOW, xlim[1])
            cols = [mode_col[(sign, n)] for n in n_group]
            for n in n_group:
                color, i = config.colors[n], mode_col[(sign, n)]
                ax.semilogy(t_td, np.abs(A_td[row, n]), color=color, lw=LW, label=rf"$n={n}$")
                ax.semilogy(t0_const, C_const[:, i], color=color, lw=LW, ls=LS_CONST)
                if C_td is not None:
                    ax.semilogy(t0_array, C_td[:, i], color=color, lw=LW, ls=LS_TD)
            ax.hlines(np.abs(A_const[row, n_group]), *xlim, colors=CONST_COLOR, alpha=CONST_ALPHA,
                      lw=CONST_LW, zorder=0)
            post = (t_td >= t_bounce) & (t_td <= t_hi)
            in_view += [np.abs(A_td[row, n_group])[:, post], np.abs(A_const[row, n_group]),
                        C_const[t0_const <= t_hi][:, cols]]
            if C_td is not None:
                in_view.append(C_td[(t0_array >= t_bounce) & (t0_array <= t_hi)][:, cols])
            mark_times(ax, meta)
            ax.set_xlim(*xlim)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, loc="upper left", frameon=False))
        axes[row, 0].set_ylim(*log_ylim(in_view, Y_PAD))

    handles = [Line2D([0], [0], color="k", lw=LW, label=r"$|\hat{A}(t)|$"),
               Line2D([0], [0], color=CONST_COLOR, alpha=CONST_ALPHA, lw=CONST_LW, label=r"$|\hat{A}_{\rm const}|$"),
               Line2D([0], [0], color="k", lw=LW, ls=LS_CONST, label=r"$|\hat{C}(t_0)|$, constant data")]
    if C_td is not None:
        handles.append(Line2D([0], [0], color="k", lw=LW, ls=LS_TD, label=r"$|\hat{C}(t_0)|$, time-dependent data"))
    axes[1, 1].legend(handles=handles, loc="lower right", fontsize=6)
    return fig


def main():
    meta, amp = load_package(PACKAGE_DIR)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    t_td, A_td = get_analytical_amplitude(amp, meta, n_max=N_MAX)
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)

    t, h_td = load_analytical_model(PACKAGE_DIR, meta=meta, data=amp)
    t_c, h_c = load_analytical_model(PACKAGE_DIR, meta=meta, data=amp, amplitudes="constant")
    t0_const = T0_ARRAY[T0_ARRAY >= t_bounce]

    print("Fitting constant-amplitude data...", flush=True)
    C_const = fit_amplitudes(t_c, h_c, MODES, t0_const, T_FIT, lambda_reg=LAMBDA_REG)
    print("Fitting time-dependent data...", flush=True)
    C_td = fit_amplitudes(t, h_td, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)

    for n in range(N_MAX + 1):
        i = MODES.index((2, 2, n, 1))
        at = lambda C, tt, x: C[np.argmin(np.abs(tt - x)), i]
        print(f"n={n}+: |A_const|={abs(A_const[0, n]):.3g}   fit const data at t0=10/20/30: "
              + " ".join(f"{at(C_const, t0_const, x):.3g}" for x in (10, 20, 30))
              + "   fit TD data: " + " ".join(f"{at(C_td, T0_ARRAY, x):.3g}" for x in (10, 20, 30)))

    save(plot(T0_ARRAY, t_td, A_td, A_const, C_const, t0_const, meta), "1b_const_fit_reg")
    save(plot(T0_ARRAY, t_td, A_td, A_const, C_const, t0_const, meta, C_td=C_td), "1c_const_vs_td_fit")


if __name__ == "__main__":
    main()
