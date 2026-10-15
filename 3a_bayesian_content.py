"""3_bayesian_content on the well-specified control: the table's constant-amplitude waveform.

Same noise level as script 3; t0 starts at the bounce, where the reconstruction begins.
"""

import numpy as np
import matplotlib.pyplot as plt

from bayesian import (resolution_noise_var, run_bgp_select, bgp_sweep, plot_mode_content,
                      plot_bgp_amplitudes)
from core import (load_package, load_analytical_model, load_constant_amplitudes,
                  peak_frame_times, mode_set, package_path)
from plotting import save

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET         = "ID00"
FIT_MODE        = "content"   # "content" (BGP per-t0) or "all" (all modes)
DATA_TYPE       = "strain"    # the constant-amplitude table holds strain amplitudes
LAMBDA_REG      = 0
N_MAX           = 7
NOISE_VAR       = None        # None -> measure it from the resolution residual
NOISE_DT_BOUNCE = 1.0         # [M] after the bounce, where that residual is sampled
NUM_SAMPLES     = 1000
T_FIT           = 100.0
MF              = 1.0
CHIF            = 0.0
INCLUDE_RETRO   = True        # only used when FIT_MODE = "all"

T0_STEP = 0.5
T0_MAX  = 40.0

Y_LIM = (1e-3, 1e8)

PACKAGE_DIR = package_path(DATASET)
NAME        = f"3a_bayesian_content_constant_{FIT_MODE}"


def main():
    meta, amp_data = load_package(PACKAGE_DIR, data_type=DATA_TYPE)
    t_bounce = peak_frame_times(meta)["t_bounce"]
    t, h22 = load_analytical_model(PACKAGE_DIR, data_type=DATA_TYPE, meta=meta,
                                   data=amp_data, amplitudes="constant")

    t0_content = np.arange(T0_STEP * np.ceil(t_bounce / T0_STEP), T0_MAX + T0_STEP / 2, T0_STEP)
    t0_array   = t0_content[:-1] if t0_content[-1] == T0_MAX else t0_content
    x_lim      = (t0_array[0], t0_array[-1])

    # Fit range from the bounce on, so no NaN samples.
    mask = (t >= max(t_bounce, t0_content[0] - 10.0)) & (t <= t0_content[-1] + T_FIT + 10.0)
    t, h22 = t[mask], h22[mask]
    assert np.all(np.isfinite(h22)), "NaNs in the masked constant-amplitude data"

    if NOISE_VAR is None:
        noise_var, shift = resolution_noise_var(PACKAGE_DIR, t_bounce + NOISE_DT_BOUNCE,
                                                N_MAX, DATA_TYPE)
        print(f"  Noise from N_int 40 vs 20 residual: {noise_var:.3e} "
              f"(alignment shift {shift} samples)", flush=True)
    else:
        noise_var = NOISE_VAR

    if FIT_MODE == "content":
        modes_list = run_bgp_select(t, h22, noise_var, t0_content, N_MAX, T_FIT, DATA_TYPE,
                                    Mf=MF, chif=CHIF)
        mode_keys = {(m[2], m[3]) for ml in modes_list for m in ml}

        def get_modes(t0):
            return modes_list[int(np.argmin(np.abs(t0_content - t0)))]

        fig = plot_mode_content(modes_list, t0_content, N_MAX, meta,
                                f"constant amplitude, {DATASET}")
        save(fig, f"{NAME}_modes", bbox_inches="tight")
        plt.close(fig)
    elif FIT_MODE == "all":
        all_modes = mode_set(N_MAX, INCLUDE_RETRO)
        mode_keys = {(m[2], m[3]) for m in all_modes}

        def get_modes(t0):
            return all_modes
    else:
        raise ValueError(f"Unknown FIT_MODE: {FIT_MODE!r}")

    bands = bgp_sweep(t, h22, get_modes, mode_keys, noise_var, t0_array, T_FIT, DATA_TYPE,
                      lambda_reg=LAMBDA_REG, num_samples=NUM_SAMPLES, Mf=MF, chif=CHIF)

    t_const, A_const = load_constant_amplitudes("constant", n_max=N_MAX, data=amp_data, meta=meta)
    fig = plot_bgp_amplitudes(t0_array, bands, t_const, A_const, meta, N_MAX, (x_lim, x_lim),
                              Y_LIM, ref_label=r"$|A_{\rm const}|$")
    save(fig, NAME, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
