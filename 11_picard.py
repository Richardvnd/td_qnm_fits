"""Discrete Picard plot (sigma_i, |u_i^H h|, |u_i^H h|/sigma_i) for the full-mode fit at T0.

Prints where the true error (u_i^H r)/sigma_i, r = h - A C_true, overtakes the solution coefficient.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import (load_package, load_analytical_model, get_analytical_amplitude,
                  load_constant_amplitudes, peak_frame_times, qnm_frequencies,
                  mode_set, package_path)
from plotting import config, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET     = "ID00"    # "ID00" or "ID10"
DATA_TYPE   = "strain"  # "strain" or "psi4"
T_FIT       = 100.0     # fitting window [M]

N_MAX         = 7
INCLUDE_RETRO = True
T0            = 20.0    # fitting start time [M] (must be >= t_bounce)

# Datasets plotted, with the label used in the colour legend.
AMPLITUDE_MODELS = {"time_dependent": "Time-dependent",
                    "constant":       "Constant"}

Y_LIM = (1e-18, 1e6)

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)


def true_coefficients(model, meta, amp_data, t0, frequencies):
    """True raw coefficients C_true = A_hat(t0) exp(-i omega t0)."""
    if model == "time_dependent":
        t, A = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
        A = A[:, :, np.argmin(np.abs(t - t0))]
    else:
        A = load_constant_amplitudes(kind=model, n_max=N_MAX, meta=meta)

    A_hat = np.array([A[0 if sign == 1 else 1, n] for _, _, n, sign in MODES])
    return A_hat * np.exp(-1j * frequencies * t0)


def picard_quantities(tau, data, t0, frequencies, C_true):
    """(sigma, |u† h|, |u† r|/sigma) at ``t0``, with r the residual against the TRUE amplitudes."""
    mask = (tau >= t0) & (tau < t0 + T_FIT) & np.isfinite(data)
    tt, dd = tau[mask], data[mask]
    a = np.exp(-1j * np.outer(tt - t0, frequencies))

    U, s, _ = np.linalg.svd(a, full_matrices=False)
    b = U.conj().T @ dd
    r = dd - a @ C_true
    return s, np.abs(b), np.abs(U.conj().T @ r)/s


def plot_picard(results):
    """Picard quantities for every dataset on one axis; ``results``: model -> output of picard()."""
    colors = {"time_dependent": config.color_td, "constant": config.color_const}

    fig, ax = plt.subplots(figsize=(config.fig_width, config.fig_height))
    fig.set_layout_engine("none")
    fig.subplots_adjust(left=0.19, right=0.97, top=0.95, bottom=0.16)

    s = next(iter(results.values()))[0]          # shared by both datasets
    idx = np.arange(1, len(s) + 1)
    ax.semilogy(idx, s, color="k", marker="o", ms=2.5, lw=0.8)

    for model in AMPLITUDE_MODELS:
        _, b, _ = results[model]
        ax.semilogy(idx, b,     color=colors[model], marker="s", ms=2.5, lw=0.8, ls="--")
        ax.semilogy(idx, b/s,   color=colors[model], marker="v", ms=2.5, lw=0.8, ls="-.")

    ax.set_xlim(0.5, len(s) + 0.5)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks(np.arange(2, len(s) + 1, 2))
    ax.set_xlabel(r"$i$")
    ax.set_ylabel("Magnitude")

    quantity_lgd = ax.legend(handles=[
        Line2D([0], [0], color="k", marker="o", ms=2.5, lw=0.8, label=r"$\sigma_i$"),
        Line2D([0], [0], color="k", marker="s", ms=2.5, lw=0.8, ls="--",
               label=r"$|u_i^\dagger h|$"),
        Line2D([0], [0], color="k", marker="v", ms=2.5, lw=0.8, ls="-.",
               label=r"$|u_i^\dagger h| / \sigma_i$"),
    ], loc="lower left", ncol=1, fontsize=6, frameon=False,
        bbox_to_anchor=(0.01, 0.01))
    ax.add_artist(quantity_lgd)

    ax.legend(handles=[Line2D([0], [0], color=colors[model], lw=0.9, label=label)
                       for model, label in AMPLITUDE_MODELS.items()],
              loc="upper right", ncol=1, fontsize=6, frameon=False,
              bbox_to_anchor=(0.99, 0.99))

    return fig


def main():
    """Build both waveforms and save the Picard plot."""
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    if T0 < t_bounce:
        raise ValueError(f"T0 must be at or after the bounce ({t_bounce:.2f} M) "
                         "— the constant-amplitude waveform is undefined before it.")

    frequencies = qnm_frequencies(MODES, 0.0, 1.0)

    results = {}
    for model in AMPLITUDE_MODELS:
        tau, data = load_analytical_model(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta,
                                          data=amp_data, amplitudes=model)
        C_true = true_coefficients(model, meta, amp_data, T0, frequencies)
        s, b, err = picard_quantities(tau, data, T0, frequencies, C_true)
        results[model] = (s, b, err)
        # Directions are meaningful only while the error stays below the coefficient.
        crossing = np.where(err > np.abs(b/s))[0]
        first = (crossing[0] + 1) if len(crossing) else len(s) + 1
        print(f"{model:>15}: kappa={s[0]/s[-1]:.2e}  max |u_i† r|/sigma_i = "
              f"{err.max():.2e}  error exceeds the coefficient at i = {first}")

    save(plot_picard(results), "11_picard")


if __name__ == "__main__":
    main()
