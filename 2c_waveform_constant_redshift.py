"""|Re h22| of the table pieces: constant-amplitude QNM sum, redshift sum, their total, the analytical
waveform and RWZ.

h_const = sum_n A_const,n e^{-i w_n t};  h_red = sum_n A_red,n(t) e^{-i w_n t} = (sum_n Red_n) e^{-t/4},
since A_red,n(t) = Red_n e^{(i w_n - 1/4) t}. Table pieces are drawn from the bounce.
"""

import numpy as np
import matplotlib.pyplot as plt

from core import (load_package, load_waveform, load_analytical_model, load_constant_amplitudes,
                  package_path, peak_frame_times, qnm_frequencies)
from plotting import config, mark_times, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET = "ID00"
N_MAX   = 7
X_LIM   = (-10.0, 57.0)   # analytical data stop at T_END_SAFE
Y_LIM   = (1e-7, 10.0)

PACKAGE_DIR = package_path(DATASET)


def main():
    meta, amp = load_package(PACKAGE_DIR)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    t, h_td = load_analytical_model(PACKAGE_DIR, meta=meta, data=amp)
    t_rwz, h_rwz = load_waveform(PACKAGE_DIR, True, meta=meta, data=amp)

    A_c = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta)
    R   = load_constant_amplitudes("redshift", n_max=N_MAX, meta=meta)
    omegas = [qnm_frequencies([(2, 2, n, s) for n in range(N_MAX + 1)]) for s in (1, -1)]
    h_const = sum(A_c[p, n] * np.exp(-1j * omegas[p][n] * t) for p in (0, 1) for n in range(N_MAX + 1))
    h_red = R.sum() * np.exp(-t / 4)
    post = t >= t_bounce

    fig, ax = plt.subplots(figsize=(config.fig_width_2, config.fig_height))
    ax.semilogy(t_rwz, np.abs(h_rwz.real), color="0.65", lw=2.2, label="RWZ")
    ax.semilogy(t, np.abs(h_td.real), color="k", lw=0.9, label=r"analytical, $\hat A(t)$")
    ax.semilogy(t[post], np.abs(h_const[post].real), color="r", lw=0.9, label="constant amplitudes")
    ax.semilogy(t[post], np.abs(h_red[post].real), color="#7b3294", lw=1.2, label="redshift modes")
    ax.semilogy(t[post], np.abs((h_const + h_red)[post].real), color="C1", lw=0.9, ls="--",
                label="constant + redshift")
    mark_times(ax, meta)
    ax.set_xlim(*X_LIM); ax.set_ylim(*Y_LIM)
    ax.set_xlabel(r"$t$ [$M$]"); ax.set_ylabel(r"$|{\rm Re}\, h_{22}|$")
    ax.legend(fontsize=7, loc="lower left", frameon=False)
    save(fig, "2c_waveform_constant_redshift")


if __name__ == "__main__":
    main()
