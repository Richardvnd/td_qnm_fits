"""Condition number of the ridge problem vs highest overtone N, for each lambda^2.

kappa = sqrt((sigma_max^2 + lambda^2) / (sigma_min^2 + lambda^2)); ridge caps it near sigma_max/lambda.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import (load_package, load_analytical_model, peak_frame_times,
                  qnm_frequencies, mode_set, package_path)
from plotting import config, lambda_label, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET     = "ID00"    # "ID00" or "ID10"
DATA_TYPE   = "strain"  # "strain" or "psi4"
T_FIT       = 100.0     # fitting window [M]

N_MAX         = 7
T0            = 20.0    # fitting start time [M] (must be >= t_bounce)
LAMBDA_LIST   = [0.0, 1e-6, 1e-4, 1e-2]     # as in 10_amplitudes_reg.py

AMPLITUDE_MODEL = "time_dependent"   # or "constant" for the exact-model control

Y_LIM = (1e0, 1e10)

PACKAGE_DIR = package_path(DATASET)


def condition_numbers(tt, t0, n_max):
    """(kappas over LAMBDA_LIST, sigma_max, sigma_min) for sample times ``tt``."""
    freqs = qnm_frequencies(mode_set(n_max), 0.0, 1.0)
    a = np.exp(-1j * np.outer(tt - t0, freqs))
    s = np.linalg.svd(a, compute_uv=False)

    # LAMBDA_LIST holds lambda^2.
    kappas = [np.sqrt((s[0]**2 + lam) / (s[-1]**2 + lam)) for lam in LAMBDA_LIST]
    return kappas, s[0], s[-1]


def plot_condition_numbers(kappas):
    """Condition number vs N; ``kappas`` maps lambda^2 -> array over N."""
    n_axis = np.arange(N_MAX + 1)
    lam_colors = config.colors2(len(LAMBDA_LIST))

    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.16)

    for i, lam in enumerate(LAMBDA_LIST):
        ax.semilogy(n_axis, kappas[lam], color=lam_colors[i],
                    marker="o", ms=2.5, lw=0.9)

    ax.set_xlim(n_axis[0] - 0.6, n_axis[-1] + 0.6)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(n_axis)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(r"$\kappa_{\lambda}$")

    ax.legend(handles=[Line2D([0], [0], color=lam_colors[i], lw=0.9,
                              label=lambda_label(lam))
                       for i, lam in enumerate(LAMBDA_LIST)],
              loc="upper left", ncol=2, fontsize=6, frameon=False,
              bbox_to_anchor=(0.01, 0.99))

    return fig


def main():
    """Compute the conditioning vs highest overtone N and save the figure."""
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    if T0 < t_bounce:
        raise ValueError(f"T0 must be at or after the bounce ({t_bounce:.2f} M).")

    tau, data = load_analytical_model(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta,
                                      data=amp_data, amplitudes=AMPLITUDE_MODEL)
    mask = (tau >= T0) & (tau < T0 + T_FIT) & np.isfinite(data)
    tt = tau[mask]

    rows = [(n_max, *condition_numbers(tt, T0, n_max))
            for n_max in range(N_MAX + 1)]
    kappas = {lam: np.array([r[1][i] for r in rows])
              for i, lam in enumerate(LAMBDA_LIST)}

    print(f"\n{AMPLITUDE_MODEL}   (t0 = {T0:g} M)   kappa_lambda")
    print(f"{'N':>6} {'sig_max':>10} {'sig_min':>10} | "
          + "".join(f"{('lam2=%g' % l):>12}" for l in LAMBDA_LIST))
    for n_max, kaps, s_max, s_min in rows:
        print(f"{2*(n_max+1):>6} {s_max:10.3e} {s_min:10.3e} | "
              + "".join(f"{k:12.3e}" for k in kaps))

    save(plot_condition_numbers(kappas), "12_condition_number")


if __name__ == "__main__":
    main()
