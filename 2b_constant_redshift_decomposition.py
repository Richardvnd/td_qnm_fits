"""Table decomposition A_n(t) = A_const,n + A_red,n(t) per overtone, and the waveform it implies.

Top: co-rotating |A_const| (red), |A_red(t)| = |Red_n| e^{(|Im w_n| - 1/4) t} (purple) and |A_const + A_red|
(dashed) against the analytical |A(t)| (black), from the bounce.
Bottom left: h_QNM = sum A_const e^{-i w t} and h_red = (sum_n Red_n) e^{-t/4} (every mode's redshift term
is Red_n e^{-t/4}), their sum, the analytical reconstruction and RWZ.
Bottom right: the 16 per-mode redshift terms |Red_n| e^{-t/4} against their sum.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from core import (load_package, load_waveform, load_analytical_model, get_analytical_amplitude,
                  load_constant_amplitudes, package_path, peak_frame_times, qnm_frequencies)
from plotting import config, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET = "ID00"
N_MAX   = 7
X_LIM   = (-10.0, 40.0)
AMP_DECADES_BELOW = 3   # amplitude panels: y floor this many decades below |A_const|
LW      = 0.9
CONST_COLOR, RED_COLOR = "r", "#7b3294"

PACKAGE_DIR = package_path(DATASET)


def main():
    meta, amp = load_package(PACKAGE_DIR)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    t_td, A_td = get_analytical_amplitude(amp, meta, n_max=N_MAX)
    t, h_td = load_analytical_model(PACKAGE_DIR, meta=meta, data=amp)
    t_rwz, h_rwz = load_waveform(PACKAGE_DIR, True, meta=meta, data=amp)

    A_c = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)
    R   = load_constant_amplitudes("redshift", n_max=N_MAX, meta=meta)
    omegas = [qnm_frequencies([(2, 2, n, s) for n in range(N_MAX + 1)]) for s in (1, -1)]

    post = t_td >= t_bounce
    tp = t_td[post]

    fig = plt.figure(figsize=(config.fig_width_2, 2.3 * config.fig_height))
    gs = fig.add_gridspec(3, 4, height_ratios=(1, 1, 1.4), hspace=0.3, wspace=0.45)

    # ── Per-overtone amplitudes (co-rotating) ──
    amp_axes = [fig.add_subplot(gs[n // 4, n % 4]) for n in range(N_MAX + 1)]
    for n, ax in enumerate(amp_axes):
        A_red = R[0, n] * np.exp(tp * (1j * omegas[0][n] - 0.25))
        ax.semilogy(t_td, np.abs(A_td[0, n]), color="k", lw=LW)
        ax.hlines(abs(A_c[0, n]), t_bounce, X_LIM[1], colors=CONST_COLOR, alpha=0.6, lw=1.2)
        ax.semilogy(tp, np.abs(A_red), color=RED_COLOR, lw=1.3)
        ax.semilogy(tp, np.abs(A_c[0, n] + A_red), color="C1", lw=LW, ls="--")
        mark_times(ax, meta, lw=0.6)
        ax.set_xlim(*X_LIM)
        in_view = (t_td >= X_LIM[0]) & (t_td <= X_LIM[1])
        y_hi = max(np.abs(A_td[0, n, in_view]).max(), np.abs(A_c[0, n] + A_red).max())
        ax.set_ylim(abs(A_c[0, n]) * 10.0**-AMP_DECADES_BELOW, y_hi * 10)
        ax.text(0.04, 0.92, rf"$n={n}$", transform=ax.transAxes, fontsize=7, va="top")
        ax.tick_params(labelsize=6)
        if n < 4:
            ax.tick_params(labelbottom=False)
    amp_axes[0].set_ylabel("Amplitude", fontsize=7)
    amp_axes[4].set_ylabel("Amplitude", fontsize=7)
    fig.legend(handles=[
        Line2D([0], [0], color="k", lw=LW, label=r"$|\hat A(t)|$"),
        Line2D([0], [0], color=CONST_COLOR, alpha=0.6, lw=1.2, label=r"$|\hat A_{\rm const}|$"),
        Line2D([0], [0], color=RED_COLOR, lw=1.3, label=r"$|\hat A_{\rm red}(t)|$"),
        Line2D([0], [0], color="C1", lw=LW, ls="--", label=r"$|\hat A_{\rm const}+\hat A_{\rm red}|$"),
    ], fontsize=7, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))

    # ── Waveform: QNM part vs redshift part ──
    mask = t >= t_bounce
    h_qnm = sum(A_c[p, n] * np.exp(-1j * omegas[p][n] * t) for p in (0, 1) for n in range(N_MAX + 1))
    h_red = R.sum() * np.exp(-t / 4)

    ax = fig.add_subplot(gs[2, :2])
    ax.semilogy(t_rwz, np.abs(h_rwz), color="0.6", lw=2.5, label=r"$|h^{\rm RWZ}|$")
    ax.semilogy(t, np.abs(h_td), color="k", lw=LW, label=r"$|h|$ (analytical $\hat A(t)$)")
    ax.semilogy(t[mask], np.abs(h_qnm[mask]), color=CONST_COLOR, lw=LW, label=r"$|h_{\rm QNM}|$")
    ax.semilogy(t[mask], np.abs(h_red[mask]), color=RED_COLOR, lw=1.3, label=r"$|h_{\rm red}|$")
    ax.semilogy(t[mask], np.abs(h_qnm[mask] + h_red[mask]), color="C1", lw=LW, ls="--",
                label=r"$|h_{\rm QNM}+h_{\rm red}|$")
    mark_times(ax, meta, lw=0.6)
    ax.set_xlim(*X_LIM); ax.set_ylim(1e-6, 10)
    ax.set_xlabel(r"$t$ [$M$]"); ax.set_ylabel("Waveform")
    ax.legend(fontsize=5.5, loc="lower left", frameon=False)

    # ── Redshift cancellation: per-mode terms vs their sum ──
    ax = fig.add_subplot(gs[2, 2:])
    for p, ls in ((0, "-"), (1, ":")):
        for n in range(N_MAX + 1):
            ax.semilogy(t[mask], abs(R[p, n]) * np.exp(-t[mask] / 4), color=config.colors[n], lw=0.7, ls=ls)
    ax.semilogy(t[mask], np.abs(h_red[mask]), color=RED_COLOR, lw=1.5)
    mark_times(ax, meta, lw=0.6)
    ax.set_xlim(*X_LIM); ax.set_ylim(1e-6, 10)
    ax.set_xlabel(r"$t$ [$M$]"); ax.tick_params(labelleft=False)
    ax.legend(handles=[
        Line2D([0], [0], color=config.colors[2], lw=0.7, label=r"$|{\rm Red}_n|\,e^{-t/4}$ (co-rot.)"),
        Line2D([0], [0], color=config.colors[2], lw=0.7, ls=":", label="(counter-rot.)"),
        Line2D([0], [0], color=RED_COLOR, lw=1.5, label=r"$|\sum_n {\rm Red}_n|\,e^{-t/4}=|h_{\rm red}|$"),
    ], fontsize=5.5, loc="lower left", frameon=False)

    print(f"sum |Red_n| = {np.abs(R).sum():.3g},  |sum Red_n| = {abs(R.sum()):.3g}")
    save(fig, "2b_constant_redshift_decomposition")


if __name__ == "__main__":
    main()
