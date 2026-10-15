"""Sense check: Fig. 7 with an extra CCE line. SXS:BBH:0305 extrapolated (dash-dot),
SXS:BBH_ExtCCE:0305 superrest (solid) and Teukode (dotted), (2,2,0,+) = 1 at NORM_T0."""

import json
import pickle
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D

from core import load_teukode_data, load_sxs_data, mode_set
from fitting import fit_amplitudes
from plotting import config, amplitude_grid, save

# ── Configuration ──────────────────────────────────────────────────────────────
SXS_ID        = "SXS:BBH:0305"
CCE_DIR       = Path("/data/rvnd2-2/CCE_data/superrest_data/SXS:BBH_ExtCCE_superrest:0305")
TEUK_DIR      = "data_all"   # private Teukode waveform
TEUK_CHIF     = 0.6921
TEUK_MF       = 1.0
TEUK_T_SHIFT  = 6.0        # [M] shift that aligns the Teukode fits with NR
LAMBDA_REG    = 1e-4
N_MAX         = 7
T0_ARRAY      = np.arange(-10.0, 40.0, 0.1)
T_FIT         = 100.0
INCLUDE_RETRO = True

NORM_T0 = 10.0   # [M] normalise (2,2,0,+) amplitude to 1 at this t0

X_LIM = (T0_ARRAY[0], T0_ARRAY[-1])
X_LIM_HIGH = (0, 20)
Y_LIM = (1e-4, 1e11)

MODES = mode_set(N_MAX, INCLUDE_RETRO)


def load_cce_data(cce_dir, l=2, m=2):
    """Superrest CCE (l, m) strain from the pickled dict, t = 0 at the peak. Returns (t, h, Mf, chif)."""
    with open(cce_dir / f"{cce_dir.name}.pickle", "rb") as f:
        data = pickle.load(f)
    with open(cce_dir / f"{cce_dir.name}_metadata.json") as f:
        meta = json.load(f)

    t, h = np.asarray(data["times"]), np.asarray(data[(l, m)])
    t = t - t[np.argmax(np.abs(h))]
    Mf   = float(meta["remnant_mass"])
    chif = float(np.linalg.norm(meta["remnant_dimensionless_spin"]))
    return t, h, Mf, chif


def plot_amplitudes(t0_array, C_nr, C_cce, C_teuk, modes):
    """Extrapolated NR (dash-dot), CCE (solid) and Teukode (dotted) on the 2x2 grid."""
    mode_col = {(sign, n): i for i, (_, _, n, sign) in enumerate(modes)}
    fig, axes, n_groups = amplitude_grid(N_MAX, row_labels=("Prograde", "Retrograde"),
                                         ylabel=r"$|\hat{C}|$", height=1.5)

    for row, sign in enumerate((1, -1)):
        for col, n_group in enumerate(n_groups):
            ax = axes[row, col]
            for n in n_group:
                idx = mode_col[(sign, n)]
                color = config.colors[n]
                ax.semilogy(t0_array, C_nr[:, idx], color=color, ls="-.", label=rf"$n={n}$")
                ax.semilogy(t0_array, C_cce[:, idx], color=color, ls="-", lw=0.8, alpha=0.7)
                ax.semilogy(t0_array, C_teuk[:, idx], color=color, ls=":")
            ax.set_xlim(*(X_LIM if col == 0 else X_LIM_HIGH))
            ax.set_ylim(*Y_LIM)
            if row == 0:
                ax.add_artist(ax.legend(fontsize=7, loc="upper left", frameon=False))

    axes[1, 1].legend(handles=[
        Line2D([0], [0], color="k", ls="-.", label="NR (extrap.)"),
        Line2D([0], [0], color="k", ls="-", lw=0.8, alpha=0.7, label="NR (CCE)"),
        Line2D([0], [0], color="k", ls=":", label="Perturbative"),
    ], loc="lower right")
    return fig


def main():
    print(f"Loading {SXS_ID}...", flush=True)
    t_nr, h22_nr, Mf, chif = load_sxs_data(SXS_ID)
    print(f"  extrap: Mf = {Mf:.6f}  chif = {chif:.6f}")

    print(f"Loading CCE ({CCE_DIR.name})...", flush=True)
    t_cce, h22_cce, Mf_cce, chif_cce = load_cce_data(CCE_DIR)
    print(f"  CCE:    Mf = {Mf_cce:.6f}  chif = {chif_cce:.6f}")
    print(f"  peak |h22|: extrap = {np.abs(h22_nr).max():.4f}  CCE = {np.abs(h22_cce).max():.4f}")

    print(f"Loading Teukode data (a={TEUK_CHIF})...", flush=True)
    t_teuk, h22_teuk = load_teukode_data(TEUK_DIR, t_shift=TEUK_T_SHIFT)

    print("Fitting extrapolated, CCE and Teukode data...", flush=True)
    kw = dict(lambda_reg=LAMBDA_REG, spherical_modes=[(2, 2)])
    C_nr   = fit_amplitudes(t_nr, h22_nr, MODES, T0_ARRAY, T_FIT, Mf=Mf, chif=chif, **kw)
    C_cce  = fit_amplitudes(t_cce, h22_cce, MODES, T0_ARRAY, T_FIT, Mf=Mf_cce, chif=chif_cce, **kw)
    C_teuk = fit_amplitudes(t_teuk, h22_teuk, MODES, T0_ARRAY, T_FIT,
                            Mf=TEUK_MF, chif=TEUK_CHIF, **kw)

    ref_col = MODES.index((2, 2, 0, 1))
    ref_idx = np.argmin(np.abs(T0_ARRAY - NORM_T0))
    C_nr   = C_nr / C_nr[ref_idx, ref_col]
    C_cce  = C_cce / C_cce[ref_idx, ref_col]
    C_teuk = C_teuk / C_teuk[ref_idx, ref_col]

    ratio = np.abs(C_cce / C_nr)
    for i, (_, _, n, sign) in enumerate(MODES):
        print(f"  n={n} sign={sign:+d}: median |C_cce/C_extrap| over t0 = {np.median(ratio[:, i]):.3f}")

    save(plot_amplitudes(T0_ARRAY, C_nr, C_cce, C_teuk, MODES), "7a_amplitudes_kerr_NR_CCE")


if __name__ == "__main__":
    main()
