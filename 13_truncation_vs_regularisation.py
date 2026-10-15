"""Truncation (N = 3, lambda^2 = 0, dashed) vs regularisation (N = 7, lambda^2 = 1e-4, dotted)."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import load_package, load_analytical_model, get_analytical_amplitude, mode_set, package_path
from fitting import fit_amplitudes
from plotting import config, lambda_label, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"

N_SHOW        = 3          # overtones plotted (one colour each)
N_TRUNC       = 3          # truncated fit: modes n <= N_TRUNC
N_FULL        = 7          # full fit: modes n <= N_FULL
LAMBDA_TRUNC  = 0.0        # ridge strength for the truncated fit
LAMBDA_FULL   = 1e-4       # ridge strength for the full fit

T0_ARRAY      = np.arange(-10.0, 40.0, 0.01)
T_FIT         = 100.0
INCLUDE_RETRO = True

TRUNC_LS = "--"
FULL_LS  = ":"

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
Y_LIM = (1e-3, 1e6)

PACKAGE_DIR = package_path(DATASET)


def plot_amplitudes(t0_array, fits, t_td, A_td, meta):
    """``fits``: list of (normalised amplitudes, modes, linestyle, label)."""
    fig, axes = plt.subplots(1, 2, figsize=(config.fig_width_2, config.fig_height * 1.1),
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.04})
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=0.17)

    for p, (sign, title) in enumerate([(1, "Co-rotating"), (-1, "Counter-rotating")]):
        ax = axes[p]
        for n in range(N_SHOW + 1):
            color = config.colors[n]
            ax.semilogy(t_td, np.abs(A_td[p, n]), color=color, label=rf"$n={n}$")
            for C_norm, modes, ls, _ in fits:
                if (2, 2, n, sign) in modes:
                    ax.semilogy(t0_array, C_norm[:, modes.index((2, 2, n, sign))], color=color, ls=ls)
        mark_times(ax, meta)
        ax.set_xlim(*X_LIM)
        ax.set_ylim(*Y_LIM)
        ax.set_xlabel(r"$t_0$ [$M$]")
        ax.set_title(title)

    axes[0].set_ylabel("Amplitude")
    axes[1].tick_params(labelleft=False)
    axes[0].add_artist(axes[0].legend(fontsize=7, loc="upper left", frameon=False))
    axes[0].legend(handles=[Line2D([0], [0], color="k", label=r"$|\hat{A}(t)|$")]
                   + [Line2D([0], [0], color="k", ls=ls, label=label) for *_, ls, label in fits],
                   fontsize=7, loc="lower right", frameon=False)
    return fig


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t, data = load_analytical_model(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta, data=amp_data)
    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_SHOW)

    fits = []
    for n_max, lam, ls in ((N_TRUNC, LAMBDA_TRUNC, TRUNC_LS), (N_FULL, LAMBDA_FULL, FULL_LS)):
        print(f"n_max={n_max}, lambda={lam:g} ...", flush=True)
        modes = mode_set(n_max, INCLUDE_RETRO)
        C_norm = fit_amplitudes(t, data, modes, T0_ARRAY, T_FIT, lambda_reg=lam)
        fits.append((C_norm, modes, ls, rf"$N={n_max}$, " + lambda_label(lam)))

    save(plot_amplitudes(T0_ARRAY, fits, t_td, A_td, meta), "13_truncation_vs_regularisation")


if __name__ == "__main__":
    main()
