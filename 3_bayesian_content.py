"""BGP mode content and amplitudes (median, 25-75% band) vs analytical TD amplitudes.

Noise is the N_int 40k vs 20k residual just after the bounce; results are cached in cache/.
"""

import hashlib
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

from bayesian import (resolution_noise_var, run_bgp_select, bgp_sweep, plot_mode_content,
                      plot_bgp_amplitudes, plot_bgp_amplitudes_low)
from core import (load_package, load_rwz, get_analytical_h22np, get_analytical_amplitude,
                  load_constant_amplitudes, peak_frame_times, mode_set, package_path)
from plotting import fit_linestyle, save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET         = "ID00"
DATA_SOURCE     = "analytical"   # "rwz" or "analytical"
FIT_MODE        = "content"      # "content" (BGP per-t0) or "all" (all modes)
DATA_TYPE       = "strain"
LAMBDA_REG      = 0
N_MAX           = 7
NOISE_VAR       = None           # None -> measure it from the resolution residual
NOISE_DT_BOUNCE = 1.0            # [M] after the bounce, where that residual is sampled
NUM_SAMPLES     = 1000
T_FIT           = 100.0
MF              = 1.0
CHIF            = 0.0
INCLUDE_RETRO   = True           # only used when FIT_MODE = "all"

T0_CONTENT = np.arange(-10.0, 40.1, 0.5)   # BGP_select grid
T0_ARRAY   = np.arange(-10.0, 40.0, 0.5)   # BGP_fit grid

CACHE_DIR       = "cache"
FORCE_RECOMPUTE = False   # True bypasses any cached BGP result

X_LIM      = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_LIM      = (1e-3, 1e8)

PACKAGE_DIR = package_path(DATASET)
FIT_LS      = fit_linestyle(DATA_SOURCE == "rwz")
NAME        = f"3_bayesian_content_{DATA_SOURCE}_{FIT_MODE}"


def _cache_path(noise_var):
    """Cache path keyed by a hash of every setting that changes the BGP result."""
    payload = (
        DATASET, DATA_SOURCE, DATA_TYPE, FIT_MODE, N_MAX, LAMBDA_REG,
        float(noise_var), NUM_SAMPLES, T_FIT, MF, CHIF, INCLUDE_RETRO,
        tuple(T0_CONTENT.tolist()), tuple(T0_ARRAY.tolist()),
    )
    digest = hashlib.sha256(repr(payload).encode()).hexdigest()[:12]
    return f"{CACHE_DIR}/3_bayesian_content_{DATASET}_{DATA_SOURCE}_{FIT_MODE}_{digest}.pkl"


def run_bgp(t, h22, noise_var):
    """BGP mode selection (if FIT_MODE='content') and sweep; returns (modes_list, bands)."""
    if FIT_MODE == "content":
        print("Running BGP mode selection...", flush=True)
        modes_list = run_bgp_select(t, h22, noise_var, T0_CONTENT, N_MAX, T_FIT, DATA_TYPE,
                                    Mf=MF, chif=CHIF)
        mode_keys = {(m[2], m[3]) for ml in modes_list for m in ml}

        def get_modes(t0):
            return modes_list[int(np.argmin(np.abs(T0_CONTENT - t0)))]
    elif FIT_MODE == "all":
        modes_list = None
        all_modes = mode_set(N_MAX, INCLUDE_RETRO)
        mode_keys = {(m[2], m[3]) for m in all_modes}

        def get_modes(t0):
            return all_modes
    else:
        raise ValueError(f"Unknown FIT_MODE: {FIT_MODE!r}")

    print("Running BGP amplitude fits...", flush=True)
    bands = bgp_sweep(t, h22, get_modes, mode_keys, noise_var, T0_ARRAY, T_FIT, DATA_TYPE,
                      lambda_reg=LAMBDA_REG, num_samples=NUM_SAMPLES, Mf=MF, chif=CHIF)
    return modes_list, bands


def main():
    print(f"Loading {DATASET} {DATA_SOURCE} waveform...", flush=True)
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)

    if DATA_SOURCE == "rwz":
        t, h22 = load_rwz(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta)
    elif DATA_SOURCE == "analytical":
        t, h_modes = get_analytical_h22np(amp_data, meta, n_max=N_MAX)
        h22 = h_modes.sum(axis=(0, 1))
    else:
        raise ValueError(f"Unknown DATA_SOURCE: {DATA_SOURCE!r}")

    # Restrict to the fit range, for speed.
    mask = (t >= min(T0_CONTENT[0], T0_ARRAY[0]) - 10.0) \
         & (t <= max(T0_CONTENT[-1], T0_ARRAY[-1]) + T_FIT + 10.0)
    t, h22 = t[mask], h22[mask]

    if NOISE_VAR is None:
        noise_t0 = peak_frame_times(meta)["t_bounce"] + NOISE_DT_BOUNCE
        noise_var, shift = resolution_noise_var(PACKAGE_DIR, noise_t0, N_MAX, DATA_TYPE)
        print(f"  Noise from N_int 40 vs 20 residual at t={noise_t0:.2f}M: {noise_var:.3e} "
              f"(alignment shift {shift} samples)", flush=True)
    else:
        noise_var = NOISE_VAR

    cache_path = _cache_path(noise_var)
    if not FORCE_RECOMPUTE and os.path.exists(cache_path):
        print(f"Loading cached BGP result from {cache_path}", flush=True)
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        modes_list = cached["modes_list"]
        bands = (cached["A_median_dict"], cached["A_low_dict"], cached["A_high_dict"])
    else:
        modes_list, bands = run_bgp(t, h22, noise_var)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump({"modes_list": modes_list, "A_median_dict": bands[0],
                         "A_low_dict": bands[1], "A_high_dict": bands[2]}, f)
        print(f"Saved BGP result to {cache_path}", flush=True)

    if modes_list is not None:
        source_label = "RWZ" if DATA_SOURCE == "rwz" else "Analytical"
        fig = plot_mode_content(modes_list, T0_CONTENT, N_MAX, meta, f"{source_label}, {DATASET}")
        save(fig, f"{NAME}_modes", bbox_inches="tight")
        plt.close(fig)

    t_td, A_td = get_analytical_amplitude(amp_data, meta, n_max=N_MAX)
    A_const = load_constant_amplitudes("constant", n_max=N_MAX, meta=meta) if DATASET == "ID00" else None
    fig = plot_bgp_amplitudes(T0_ARRAY, bands, t_td, A_td, meta, N_MAX,
                              (X_LIM, X_LIM_HIGH), Y_LIM, fit_ls=FIT_LS, A_const=A_const)
    save(fig, NAME, bbox_inches="tight")
    plt.close(fig)

    fig = plot_bgp_amplitudes_low(T0_ARRAY, bands, t_td, A_td, meta, N_MAX, X_LIM, Y_LIM,
                                  fit_ls=FIT_LS, A_const=A_const)
    save(fig, f"{NAME}_low", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
