"""Single-mode test: each analytical overtone waveform fitted with only that mode.

Insets: relative residual |C - A| / |A| against t0.
"""

import numpy as np
from matplotlib.ticker import LogLocator

from core import load_package, get_analytical_amplitude, get_analytical_h22np, package_path
from fitting import fit_amplitudes
from plotting import config, log_ylim, mark_times, overtone_stack, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET    = "ID00"
DATA_TYPE  = "strain"
LAMBDA_REG = 1e-4
N_MAX      = 7
T0_ARRAY   = np.arange(-10.0, 40.0, 0.1)
T_FIT      = 100.0

X_LIM      = (T0_ARRAY[0], T0_ARRAY[-1])   # top block (n = 0..3)
X_LIM_HIGH = (0, 20.0)                      # bottom block (n = 4..7)
Y_LOG_PAD  = 0.3    # log-range padding of the n = 0, 1 rows
INSET_LOW  = [0.6, 0.08, 0.38, 0.4]   # residual inset, axes fraction (bottom right)
INSET_HIGH = [0.6, 0.52, 0.38, 0.4]   # top right, for counter-rotating n = 0

PACKAGE_DIR = package_path(DATASET)


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t_td, A_td    = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
    t_wf, h_modes = get_analytical_h22np(amp_data, meta, n_max=N_MAX)

    fig, axes, grid_row, mid = overtone_stack(N_MAX, bottom=0.05)
    fig.set_size_inches(fig.get_figwidth(), fig.get_figheight() * 1.1)

    for n in range(N_MAX + 1):
        color = config.colors[n]
        xlim = X_LIM if n < mid else X_LIM_HIGH
        mask_td = (t_td >= xlim[0]) & (t_td <= xlim[1])
        mask_t0 = (T0_ARRAY >= xlim[0]) & (T0_ARRAY <= xlim[1])

        # Fit both columns first so the row's y-limits cover both panels.
        A_vals = [np.abs(A_td[p, n])[mask_td] for p in (0, 1)]
        C_vals = [fit_amplitudes(t_wf, h_modes[p, n], [(2, 2, n, sign)], T0_ARRAY, T_FIT,
                                 lambda_reg=LAMBDA_REG)[:, 0]
                  for p, sign in enumerate((1, -1))]
        row_ylim = log_ylim(A_vals + [C[mask_t0] for C in C_vals],
                            Y_LOG_PAD if n in (0, 1) else 0.0)

        for col in range(2):
            ax = axes[grid_row(n), col]
            ax.semilogy(t_td[mask_td], A_vals[col], color=color, lw=0.6)
            ax.semilogy(T0_ARRAY, C_vals[col], color=color, lw=0.6, ls="--")
            A_t0 = np.interp(T0_ARRAY, t_td, np.abs(A_td[col, n]))
            inset = ax.inset_axes(INSET_HIGH if (n, col) == (0, 1) else INSET_LOW)
            inset.semilogy(T0_ARRAY[mask_t0], np.abs(C_vals[col] - A_t0)[mask_t0] / A_t0[mask_t0],
                           color=color, lw=0.5)
            mark_times(inset, meta, lw=0.4)
            inset.set_xlim(*xlim)
            inset.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
            inset.tick_params(labelbottom=False, labelsize=5, length=1.5)
            mark_times(ax, meta, lw=0.5)
            ax.set_xlim(*xlim)
            ax.set_ylim(*row_ylim)
            ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=4))
            ax.text(0.06, 0.94, rf"$n={n}$", transform=ax.transAxes, va="top", ha="left")
            if col == 0:
                ax.set_ylabel(r"$|\hat{C}|$")
            else:
                ax.tick_params(labelleft=False)
            if n not in (mid - 1, N_MAX):
                ax.tick_params(labelbottom=False)

    axes[0, 0].set_title("Co-rotating")
    axes[0, 1].set_title("Counter-rotating")
    for col in range(2):
        axes[grid_row(N_MAX), col].set_xlabel(r"$t_0$ [$M$]")

    save(fig, "16_single_amplitudes")


if __name__ == "__main__":
    main()
