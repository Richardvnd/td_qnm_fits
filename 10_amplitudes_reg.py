"""Least-squares amplitude sweep for a range of ridge strengths lambda^2, one overtone per panel."""

import numpy as np

from core import load_package, load_waveform, get_analytical_amplitude, mode_set, package_path
from fitting import fit_amplitudes
from plotting import lambda_label, plot_sweep_stack, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
USE_RWZ       = False
T_FIT         = 100.0      # fixed fitting window [M]

N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.01)
LAMBDA_LIST   = [0.0, 1e-6, 1e-4, 1e-2]   # ridge strengths (lambda^2)
INCLUDE_RETRO = True

X_LIM      = (T0_ARRAY[0], T0_ARRAY[-1])   # top block (n = 0..3)
X_LIM_HIGH = (0, 22.0)                      # bottom block (n = 4..7)
Y_LOG_PAD  = 0.08   # log-range padding of the n = 0, 1 rows

PACKAGE_DIR = package_path(DATASET)
MODES       = mode_set(N_MAX, INCLUDE_RETRO)


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t, data = load_waveform(PACKAGE_DIR, USE_RWZ, DATA_TYPE, meta=meta, data=amp_data)
    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)

    C_norm_by_lam = []
    for lam in LAMBDA_LIST:
        print(f"lambda={lam:g} ...", flush=True)
        C_norm_by_lam.append(fit_amplitudes(t, data, MODES, T0_ARRAY, T_FIT, lambda_reg=lam))

    fig = plot_sweep_stack(T0_ARRAY, C_norm_by_lam, [lambda_label(lam) for lam in LAMBDA_LIST],
                           t_td, A_td, meta, MODES, N_MAX, X_LIM, X_LIM_HIGH, Y_LOG_PAD,
                           legend_ncol=5)
    save(fig, "10_amplitudes_reg")


if __name__ == "__main__":
    main()
