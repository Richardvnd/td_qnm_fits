"""Least-squares amplitude sweep for a range of fitting windows T, one overtone per panel."""

import numpy as np

from core import load_package, load_waveform, get_analytical_amplitude, mode_set, package_path
from fitting import fit_amplitudes
from plotting import plot_sweep_stack, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET       = "ID00"     # "ID00" or "ID10"
DATA_TYPE     = "strain"   # "strain" or "psi4"
USE_RWZ       = False
LAMBDA_REG    = 1e-4       # ridge regularisation (0.0 = none)

N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.01)
T_LIST        = [1, 5, 10, 15, 20, 50, 100]   # fitting windows [M]
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

    C_norm_by_T = []
    for T in T_LIST:
        print(f"T={T} ...", flush=True)
        C_norm_by_T.append(fit_amplitudes(t, data, MODES, T0_ARRAY, T, lambda_reg=LAMBDA_REG))

    fig = plot_sweep_stack(T0_ARRAY, C_norm_by_T, [rf"$T={T}\,M$" for T in T_LIST],
                           t_td, A_td, meta, MODES, N_MAX, X_LIM, X_LIM_HIGH, Y_LOG_PAD,
                           legend_ncol=4)
    save(fig, "9_amplitudes_T")


if __name__ == "__main__":
    main()
