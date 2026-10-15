"""Raw (not decay-corrected) fitted |C(t0)| vs per-mode |h_22np(t)|, for RWZ and analytical data.

Overtones are offset by OVERTONE_SEP; the black segment is the redshift decay e^{-kappa t}.
"""

import numpy as np
from matplotlib.lines import Line2D

from core import (load_package, load_rwz, load_analytical_model, get_analytical_h22np,
                  mode_set, package_path)
from fitting import run_amplitude_sweep
from plotting import config, amplitude_grid, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
LAMBDA_REG    = 1e-4       # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(0.0, 40.0, 0.01)
T_FIT         = 100.0
INCLUDE_RETRO = True

KAPPA = 1.0 / 4.0          # horizon surface gravity, M = 1

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
Y_LIM = (1e-5, 1e10)

SEG_X        = 13.0        # [M] centre of the redshift-mode reference segment
SEG_WIDTH    = 8.0         # [M] its length
OVERTONE_SEP = 100.0       # successive overtones within a panel are scaled by this

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)


def plot_amplitudes(t0_array, C_rwz, C_ana, t_wf, h_modes, meta, modes):
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}
    fig, axes, n_groups = amplitude_grid(N_MAX)

    # Redshift-mode reference segment, near the top of each panel.
    y_lo, y_hi = np.log10(Y_LIM[0]), np.log10(Y_LIM[1])
    y_seg_start = 10 ** (y_lo + 0.88 * (y_hi - y_lo))
    t_seg = np.array([SEG_X - SEG_WIDTH / 2, SEG_X + SEG_WIDTH / 2])
    y_seg = y_seg_start * np.exp(-KAPPA * (t_seg - t_seg[0]))
    y_seg_mid = y_seg_start * np.exp(-KAPPA * SEG_WIDTH / 2)

    for row, sign in enumerate((1, -1)):
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            for i, n in enumerate(n_group):
                color = config.colors[n]
                scale = OVERTONE_SEP ** i
                exponent = i * np.log10(OVERTONE_SEP)
                label = rf"$n={n}$" if i == 0 else rf"$n={n}\ (\times 10^{{{exponent:.0f}}})$"
                ax.semilogy(t_wf, scale * np.abs(h_modes[row, n]), color=color, alpha=0.3, label=label)
                if (sign, n) in mode_col:
                    idx = mode_col[(sign, n)]
                    ax.semilogy(t0_array, scale * np.abs(C_rwz[:, idx]), color=color, ls=":")
                    ax.semilogy(t0_array, scale * np.abs(C_ana[:, idx]), color=color, ls="--")

            ax.semilogy(t_seg, y_seg, color="k", ls=":", lw=1.0)
            ax.text(SEG_X, y_seg_mid * 2.5, r"$e^{-\kappa t}$", ha="center", va="bottom")
            mark_times(ax, meta)
            ax.set_xlim(*X_LIM)
            ax.set_ylim(*Y_LIM)
            if row == 0:
                legend = ax.legend(loc="upper right", frameon=False, fontsize=6)
                for line in legend.get_lines():
                    line.set_alpha(1.0)
                ax.add_artist(legend)

    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="k", label=r"$|A(t)|$"),
        Line2D([0], [0], color="k", ls=":", label=r"$|C(t_0)|$ (perturbative)"),
        Line2D([0], [0], color="k", ls="--", label=r"$|C(t_0)|$ (analytical)"),
    ], loc="upper right", fontsize=6)
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t_rwz, data_rwz = load_rwz(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta)
    t_ana, data_ana = load_analytical_model(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta, data=amp_data)

    t_wf, h_modes = get_analytical_h22np(amp_data, meta, n_max=N_MAX)
    C_rwz, *_ = run_amplitude_sweep(t_rwz, data_rwz, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)
    C_ana, *_ = run_amplitude_sweep(t_ana, data_ana, MODES, T0_ARRAY, T_FIT, lambda_reg=LAMBDA_REG)

    save(plot_amplitudes(T0_ARRAY, C_rwz, C_ana, t_wf, h_modes, meta, MODES), "6_redshift")


if __name__ == "__main__":
    main()
